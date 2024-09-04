# %%
import sentencepiece as spm
from gemma import params as params_lib
from gemma import transformer as transformer_lib

import os
import re
import kagglehub
from einops import rearrange
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import jax_extra
from train import Config, Model, training_step, State
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
from omegaconf import OmegaConf
from input_loader import HuggingFaceDataParams, HuggingFaceDataLoader, TokenBatchParams, TokenBatch
from jax.sharding import Mesh
from jax.experimental import mesh_utils
os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=8"  # Use 8 CPU devices
)


kagglehub.login()
# %%
kagglehub.model_download("google/gemma-2/flax/gemma2-2b")

# %%
model_dir = "/Users/clankur/.cache/kagglehub/models"
gemma_dir = f"{model_dir}/google/gemma-2/flax/gemma2-2b/1"
checkpoint_dir = f"{model_dir}/google/gemma-2/flax/gemma2-2b/1/gemma2-2b"
# %%
params = checkpoints.restore_checkpoint(checkpoint_dir, target=None)
params = {key: weights.popitem()[1] for (key, weights) in params.items()}
# %%
config_name = "arxiv_v4-32_2b_gemma2.yaml"
GlobalHydra.instance().clear()
with initialize(version_base=None, config_path="configs"):
    cfg = compose(config_name=f"{config_name}")

config: Config = jax_extra.make_dataclass_from_dict(
    Config, OmegaConf.to_container(cfg, resolve=True)
)
config
# %%
c_model = config.model
seqax_model = {
    "embed": params["transformer/embedder"],
    "unembed": params["transformer/embedder"],
    "ln1": jnp.zeros((c_model.layers, c_model.d_model)),
    "ln2": jnp.zeros((c_model.layers, c_model.d_model)),
    "post_attn_ln": jnp.zeros((c_model.layers, c_model.d_model)),
    "post_ffn_ln": jnp.zeros((c_model.layers, c_model.d_model)),
    "w_q": jnp.zeros((c_model.layers, c_model.d_model, c_model.n_q_per_kv, c_model.n_kv, c_model.d_head)),
    "w_kv": jnp.zeros((c_model.layers, 2, c_model.d_model, c_model.n_kv, c_model.d_head)),
    "w_o": jnp.zeros((c_model.layers, c_model.d_model, c_model.n_q_per_kv, c_model.n_kv, c_model.d_head)),
    "w_gate": jnp.zeros((c_model.layers, c_model.d_model, c_model.d_ff)),
    "w_up": jnp.zeros((c_model.layers, c_model.d_model, c_model.d_ff)),
    "w_down": jnp.zeros((c_model.layers, c_model.d_model, c_model.d_ff)),
    "final_layer_norm": params["transformer/final_norm"],
}

# %%
for name, weights in params.items():
    if "layer" in name:
        layer_match = re.search(r"layer_(\d+)", name)
        layer_idx = int(layer_match.group(1))

        if "pre_attention_norm" in name:
            seqax_model["ln1"] = seqax_model["ln1"].at[layer_idx].set(weights)
        elif "pre_ffw_norm" in name:
            seqax_model["ln2"] = seqax_model["ln2"].at[layer_idx].set(weights)
        elif "attn/kv_einsum" in name:
            seqax_model["w_kv"] = seqax_model["w_kv"].at[layer_idx].set(
                rearrange(
                    weights,
                    "kv n_kv d_model d_head -> kv d_model n_kv d_head",
                )
            )
        elif "attn/q_einsum" in name:
            weights = rearrange(
                weights,
                "(n_q_per_kv n_kv) d_model d_head -> d_model n_q_per_kv n_kv d_head",
                n_q_per_kv=c_model.n_q_per_kv,
                n_kv=c_model.n_kv,
            )
            seqax_model["w_q"] = seqax_model["w_q"].at[layer_idx].set(
                weights
            )
        elif "attn/attn_vec_einsum" in name:
            weights = rearrange(
                weights,
                "(n_q_per_kv n_kv) d_head d_model  -> d_model n_q_per_kv n_kv d_head",
                n_q_per_kv=c_model.n_q_per_kv,
                n_kv=c_model.n_kv,
            )
            seqax_model["w_o"] = seqax_model["w_o"].at[layer_idx].set(
                weights
            )
        elif "mlp/gating_einsum" in name:
            seqax_model["w_gate"] = seqax_model["w_gate"].at[layer_idx].set(
                weights[0, :, :]
            )
            seqax_model["w_up"] = seqax_model["w_up"].at[layer_idx].set(
                weights[1, :, :]
            )
        elif "mlp/linear" in name:
            weights = rearrange(weights, "d_ff d_model ->d_model d_ff")
            seqax_model["w_down"] = seqax_model["w_down"].at[layer_idx].set(
                weights
            )
        elif "post_attention_norm" in name:
            seqax_model["post_attn_ln"] = seqax_model["post_attn_ln"].at[layer_idx].set(
                weights
            )
        elif "post_ffw_norm" in name:
            seqax_model["post_ffn_ln"] = seqax_model["post_ffn_ln"].at[layer_idx].set(
                weights
            )
        else:
            print(name, weights.shape)
    else:
        print(name, weights.shape)

# %%
model_weights = Model(
    embed=seqax_model["embed"].astype(jnp.float32),
    unembed=seqax_model["unembed"].astype(jnp.float32),
    ln1=seqax_model["ln1"].astype(jnp.float32),
    ln2=seqax_model["ln2"].astype(jnp.float32),
    post_attn_ln=seqax_model["post_attn_ln"].astype(jnp.float32),
    post_ffn_ln=seqax_model["post_ffn_ln"].astype(jnp.float32),
    w_q=seqax_model["w_q"].astype(jnp.float32),
    w_kv=seqax_model["w_kv"].astype(jnp.float32),
    w_o=seqax_model["w_o"].astype(jnp.float32),
    w_gate=seqax_model["w_gate"].astype(jnp.float32),
    w_up=seqax_model["w_up"].astype(jnp.float32),
    w_down=seqax_model["w_down"].astype(jnp.float32),
    final_layer_norm=seqax_model["final_layer_norm"].astype(jnp.float32),
)
# %%
adam_mu = jax.tree.map(lambda p: p * 0.0, model_weights)
adam_nu = jax.tree.map(lambda p: p * 0.0, model_weights)
state = State(weights=model_weights, adam_mu=adam_mu, adam_nu=adam_nu)

# %%
batch_size = 2
seq_length = 4  # tokens.targets.shape[-1]


# %%
with Mesh(
    mesh_utils.create_device_mesh(
        [1, 1], jax.devices()[:1]),
    ("d", "t"),
):
    dataloader_params = HuggingFaceDataParams(
        path="bigcode/starcoderdata",
        tokenizer="google/gemma-2b",
        num_workers=0,
        sequences_packed_per_batch=2
    )
    token_batch_params = TokenBatchParams(
        len=seq_length,
        batch=batch_size
    )
    loader = HuggingFaceDataLoader(
        split="train", config=dataloader_params, token_batch_params=token_batch_params)

    tokens: TokenBatch = loader.load(0)
    c_training_step = training_step.lower(
        state, jnp.uint32(0), config.model, config.training, tokens
    ).compile()

    state, output = c_training_step(
        state, jnp.uint32(0), tokens)


# %%
params = params_lib.load_and_format_params(checkpoint_dir)
# %%
gemma2_config = transformer_lib.TransformerConfig.gemma2_2b(1024)
gemma2_config = gemma2_config.from_params(params=params)
transformer = transformer_lib.Transformer(gemma2_config)
# %%
dummy_input = jnp.pad(tokens.targets[:, :-1], pad_width=((0, 0), (1, 0)))
dummy_positions = jnp.arange(seq_length)[None, :]
dummy_attention_mask = jnp.ones((batch_size, seq_length, seq_length))
# %%
logits, cache = transformer.apply(
    {'params': params["transformer"]},
    dummy_input,
    dummy_positions,
    None,
    dummy_attention_mask,
)

# %%
