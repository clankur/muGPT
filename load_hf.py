# %%
import jax
import jax.numpy as jnp
from transformers import AutoModelForCausalLM
from einops import rearrange
from omegaconf import OmegaConf
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import jax_extra
from train import Config, Model, training_step, State
from input_loader import get_loader
from jax.experimental import mesh_utils
from jax.sharding import Mesh
import training_io


# %%
def load_llama(hf_model, config_name):
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name=f"{config_name}")

    config: Config = jax_extra.make_dataclass_from_dict(
        Config, OmegaConf.to_container(cfg, resolve=True)
    )
    pt_state_dict = hf_model.state_dict()
    jax_params = jax.tree_util.tree_map(
        jnp.array, {k: v.numpy() for k, v in pt_state_dict.items()}
    )

    c_model = config.model
    jax_model = {
        "embed": jnp.zeros((c_model.vocab, c_model.d_model)),
        "unembed": jnp.zeros((c_model.vocab, c_model.d_model)),
        "ln1": jnp.zeros((c_model.layers, c_model.d_model)),
        "ln2": jnp.zeros((c_model.layers, c_model.d_model)),
        "w_q": jnp.zeros(
            (
                c_model.layers,
                c_model.d_model,
                c_model.n_q_per_kv,
                c_model.n_kv,
                c_model.d_head,
            )
        ),
        "w_kv": jnp.zeros(
            (c_model.layers, 2, c_model.d_model, c_model.n_kv, c_model.d_head)
        ),
        "w_o": jnp.zeros(
            (
                c_model.layers,
                c_model.d_model,
                c_model.n_q_per_kv,
                c_model.n_kv,
                c_model.d_head,
            )
        ),
        "w_gate": jnp.zeros((c_model.layers, c_model.d_model, c_model.d_ff)),
        "w_up": jnp.zeros((c_model.layers, c_model.d_model, c_model.d_ff)),
        "w_down": jnp.zeros((c_model.layers, c_model.d_model, c_model.d_ff)),
        "final_layer_norm": jnp.zeros((c_model.d_model)),
    }

    # Map Hugging Face parameters to JAX model structure
    for key, value in jax_params.items():
        if key == "model.embed_tokens.weight":
            if value.shape != jax_model["embed"].shape:
                raise ValueError(
                    f"shapes jax_model shape {jax_model['embed'].shape} != {value.shape}"
                )
            jax_model["embed"] = value
        elif key == "lm_head.weight":
            jax_model["unembed"] = value
        elif key.endswith("input_layernorm.weight"):
            layer = int(key.split(".")[2])
            jax_model["ln1"] = jax_model["ln1"].at[layer].set(value)
        elif key.endswith("post_attention_layernorm.weight"):
            layer = int(key.split(".")[2])
            jax_model["ln2"] = jax_model["ln2"].at[layer].set(value)
        elif key.endswith("self_attn.q_proj.weight"):
            print(value.shape)
            layer = int(key.split(".")[2])
            reshaped = rearrange(
                value,
                "d_model (q_per_kv n_kv d_head) -> d_model q_per_kv n_kv d_head",
                q_per_kv=c_model.n_q_per_kv,
                n_kv=c_model.n_kv,
                d_head=c_model.d_head,
            )
            jax_model["w_q"] = jax_model["w_q"].at[layer].set(reshaped)
        elif key.endswith("self_attn.k_proj.weight"):
            layer = int(key.split(".")[2])
            reshaped = rearrange(
                value,
                "(n_kv d_head) d_model -> d_model n_kv d_head",
                n_kv=c_model.n_kv,
                d_head=c_model.d_head,
            )
            jax_model["w_kv"] = jax_model["w_kv"].at[layer, 0].set(reshaped)
        elif key.endswith("self_attn.v_proj.weight"):
            layer = int(key.split(".")[2])
            reshaped = rearrange(
                value,
                "(n_kv d_head) d_model -> d_model n_kv d_head",
                n_kv=c_model.n_kv,
                d_head=c_model.d_head,
            )
            jax_model["w_kv"] = jax_model["w_kv"].at[layer, 1].set(reshaped)
        elif key.endswith("self_attn.o_proj.weight"):
            layer = int(key.split(".")[2])
            reshaped = rearrange(
                value,
                "d_model (q_per_kv n_kv d_head) -> d_model q_per_kv n_kv d_head",
                q_per_kv=c_model.n_q_per_kv,
                n_kv=c_model.n_kv,
                d_head=c_model.d_head,
            )
            jax_model["w_o"] = jax_model["w_o"].at[layer].set(reshaped)
        elif key.endswith("mlp.gate_proj.weight"):
            layer = int(key.split(".")[2])
            reshaped = rearrange(value, "d_model d_ff -> d_ff d_model")
            jax_model["w_gate"] = jax_model["w_gate"].at[layer].set(reshaped)
        elif key.endswith("mlp.up_proj.weight"):
            layer = int(key.split(".")[2])
            reshaped = rearrange(value, "d_model d_ff -> d_ff d_model")
            jax_model["w_up"] = jax_model["w_up"].at[layer].set(reshaped)
        elif key.endswith("mlp.down_proj.weight"):
            layer = int(key.split(".")[2])
            jax_model["w_down"] = jax_model["w_down"].at[layer].set(value)
        elif key == "model.norm.weight":
            jax_model["final_layer_norm"] = value
        else:
            print(f"key {key} not captured in model")

        model_weights = Model(
            embed=jax_model["embed"],
            unembed=jax_model["unembed"],
            ln1=jax_model["ln1"],
            ln2=jax_model["ln2"],
            w_q=jax_model["w_q"],
            w_kv=jax_model["w_kv"],
            w_o=jax_model["w_o"],
            w_gate=jax_model["w_gate"],
            w_up=jax_model["w_up"],
            w_down=jax_model["w_down"],
            final_layer_norm=jax_model["final_layer_norm"],
        )
    adam_mu = jax.tree.map(lambda p: p * 0.0, model_weights)
    adam_nu = jax.tree.map(lambda p: p * 0.0, model_weights)
    state = State(weights=model_weights, adam_mu=adam_mu, adam_nu=adam_nu)

    return state, config


# %%
model_name = "TinyLlama/TinyLlama_v1.1"
hf_model = AutoModelForCausalLM.from_pretrained(model_name)
config_name = "arxiv_v4-32_1b_llama.yaml"
state, config = load_llama(hf_model, config_name)
print("loaded model")
logger = None
# %%
# %% [markdown]
# ### TODO:
# - evaluate loss, perplexity
# - compare hf model to jax model
# %%
with Mesh(
    mesh_utils.create_device_mesh([config.mesh.d, config.mesh.t], jax.devices()),
    ("d", "t"),
):

    loader = get_loader("train", config.training_data, config.training.tokens)
    c_training_step = training_step.lower(
        state, jnp.uint32(0), config.model, config.training, loader.load(0)
    ).compile()

    print("forward pass")
    step = 0
    batch = loader.load(step)
    print(batch.targets)
    state, output = c_training_step(state, jnp.uint32(step), batch)
    training_io.log(step, logger, output)

    # %%
    hf_model(**batch.targets)
