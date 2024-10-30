# %%
import numpy as np
import torch
import jax
from jax import lax
import jax.numpy as jnp
from dlpack import asdlpack
from einops import rearrange
import json
import os
from train import Hparams, BaseWidths, Model, State
from typing import Tuple
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from shardlib.shardtypes import u32, bool_, f32, make_shardings

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=8"  # Use 8 CPU devices
)


# %%
def load_model_weights_and_hparams(model_path: str) -> Tuple[dict, Hparams]:
    with open(f"{model_path}/params.json", "r") as f:
        params = json.load(f)
    weights = torch.load(
        f"{model_path}/consolidated.00.pth", map_location=torch.device("cpu")
    )

    hidden_dim = int(8 * params["dim"] // 3 * params["ffn_dim_multiplier"])
    hidden_dim = params["multiple_of"] * (
        (hidden_dim + params["multiple_of"] - 1) // params["multiple_of"]
    )

    base = BaseWidths(
        d_model=params["dim"],
        n_q_per_kv=params["n_heads"] // params["n_kv_heads"],
        n_kv=params["n_kv_heads"],
        d_head=params["dim"] // params["n_heads"],
        d_ff=hidden_dim,
    )

    h = Hparams(
        d_model=params["dim"],
        n_q_per_kv=params["n_heads"] // params["n_kv_heads"],
        n_kv=params["n_kv_heads"],
        d_head=params["dim"] // params["n_heads"],
        layers=params["n_layers"],
        vocab=params["vocab_size"],
        d_ff=hidden_dim,
        rope_max_timescale=params["rope_theta"],
        norm_eps=params["norm_eps"],
        use_scaled_rope=params["use_scaled_rope"],
        # default other parameters from exp scaling/muP
        base=base,
        a_attn=1.0,
        a_output=1.0,
        zero_queries=False,
        zero_unembed=False,
        parameterization="sp",
        fully_aligned=False,
        gamma_embed=1.0,
        gamma_hidden=1.0,
        gamma_unembed=1.0,
    )
    return weights, h


# %%
def load_llama(weights: dict, h: Hparams) -> Model:
    pre_attention_norms = []
    pre_ffw_norms = []
    attn_qs = []
    attn_kvs = []
    attn_os = []
    mlp_gates = []
    mlp_ups = []
    mlp_downs = []

    embed = jnp.from_dlpack(asdlpack(weights["tok_embeddings.weight"].float()))
    unembed = jnp.from_dlpack(asdlpack(weights["tok_embeddings.weight"].float()))
    final_norm = jnp.from_dlpack(asdlpack(weights["norm.weight"].float()))

    for layer_id in range(h.layers):
        # norms
        ln1 = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.attention_norm.weight"].float())
        )
        ln2 = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.ffn_norm.weight"].float())
        )
        pre_attention_norms.append(ln1)
        pre_ffw_norms.append(ln2)

        # attention weights
        w_q = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.attention.wq.weight"].float()),
        )

        w_q = rearrange(
            w_q,
            "(n_kv n_q_per_kv d_head) d_model  -> n_kv n_q_per_kv d_head d_model",
            n_q_per_kv=h.n_q_per_kv,
            n_kv=h.n_kv,
            d_head=h.d_head,
        )  # n_kv n_q_per_kv H_dim M_dim

        attn_qs.append(w_q)

        w_k = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.attention.wk.weight"].float()),
        )

        w_k = rearrange(
            w_k,
            "(n_kv d_head) d_model -> d_model n_kv d_head",
            n_kv=h.n_kv,
            d_head=h.d_head,
        )  # M_dim n_kv H_dim

        w_v = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.attention.wv.weight"].float()),
        )

        w_v = rearrange(
            w_v,
            "(n_kv d_head) d_model -> d_model n_kv d_head",
            n_kv=h.n_kv,
            d_head=h.d_head,
        )  # M_dim n_kv H_dim
        w_kv = jnp.stack([w_k, w_v], axis=0)
        attn_kvs.append(w_kv)

        w_o = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.attention.wo.weight"].float()),
        )
        w_o = rearrange(
            w_o,
            "d_model (n_kv n_q_per_kv d_head) -> d_model n_kv n_q_per_kv d_head",
            n_q_per_kv=h.n_q_per_kv,
            n_kv=h.n_kv,
            d_head=h.d_head,
        )  # "M_dim n_kv n_q_per_kv H_dim"
        attn_os.append(w_o)

        # mlp
        w_gate = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.feed_forward.w1.weight"].float()),
        )
        w_gate = rearrange(w_gate, "d_ff d_model -> d_model d_ff")
        mlp_gates.append(w_gate)

        w_up = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.feed_forward.w3.weight"].float()),
        )
        w_up = rearrange(w_up, "d_ff d_model -> d_model d_ff")
        mlp_ups.append(w_up)

        w_down = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.feed_forward.w2.weight"].float()),
        )
        mlp_downs.append(w_down)

    pre_attention_norms = jnp.stack(pre_attention_norms, axis=0)
    pre_ffw_norms = jnp.stack(pre_ffw_norms, axis=0)
    attn_qs = jnp.stack(attn_qs, axis=0)
    attn_kvs = jnp.stack(attn_kvs, axis=0)
    attn_os = jnp.stack(attn_os, axis=0)
    mlp_gates = jnp.stack(mlp_gates, axis=0)
    mlp_ups = jnp.stack(mlp_ups, axis=0)
    mlp_downs = jnp.stack(mlp_downs, axis=0)

    model = Model(
        embed=embed,
        unembed=unembed,
        ln1=pre_attention_norms,
        ln2=pre_ffw_norms,
        w_q=attn_qs,
        w_kv=attn_kvs,
        w_o=attn_os,
        w_gate=mlp_gates,
        w_up=mlp_ups,
        w_down=mlp_downs,
        final_layer_norm=final_norm,
    )
    return model


# %%
model_path = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/")
llama_weights, h = load_model_weights_and_hparams(model_path)
model = load_llama(llama_weights, h)


# %%
with Mesh(
    mesh_utils.create_device_mesh([1, 1], jax.devices()[:1]),
    ("d", "t"),
):
    batch_size = 2
    seq_length = 16

    model = jax.tree.map(lax.with_sharding_constraint, model, make_shardings(Model))

    shape = (batch_size, seq_length)
    inputs: u32[b"batch/d len"] = jnp.zeros(shape, dtype=jnp.uint32)
    seq_starts: bool_[b"batch/d len"] = jnp.zeros(shape, dtype=bool)
    logits: f32[b"batch/d len V/t"] = model.forward_pass(h, inputs, seq_starts)

# %%
