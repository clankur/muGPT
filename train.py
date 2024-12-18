"""Main training loop, including the model, loss function, and optimizer."""

import dataclasses
from jax.tree_util import tree_leaves
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from clearml import Task
import training_io
from jax_extra import fold_in_str, explicit_activation_checkpointing, save_for_backward
import jax_extra
import einops
import shardlib.shardops as shardops
from shardlib.shardtypes import bf16, bool_, f32, pytree_dataclass, u32, make_shardings
from input_loader import (
    FlatTokensParams,
    HuggingFaceDataParams,
    SyntheticDataParams,
    TokenBatch,
    TokenBatchParams,
    get_loader,
)
import math
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jax import Array, lax
import jax
from dataclasses import dataclass
from typeguard import typechecked
import hydra
from typing import Any, Optional, Tuple, Union
from functools import cached_property, partial
from collections import defaultdict
import datetime
import gcsfs  # Needed for clearml setup
import shardlib.shardtypes as shardtypes
import operator
import os
import time
import subprocess
import signal
from collections import namedtuple
import env

env.set_variables()

shardtypes.register_with_typeguard()


P = PartitionSpec

PRNGKey = Any


@dataclass(frozen=True)
class BaseWidths:
    d_model: int
    n_q_per_kv: int
    n_kv: int
    d_head: int
    d_ff: int


@dataclass(frozen=True)
class Hparams:
    d_model: int
    d_hidden: int
    kernel_size: int
    groups: int
    dropout: float
    layers: int
    vocab: int


def get_parameterization(style: str, fully_aligned: bool = True):
    Parameterization = namedtuple(
        "Parameterization",
        [
            "embed_init_var",
            "embed_param_mult",
            "embed_lr",
            "embed_grad",
            "hidden_init_var",
            "hidden_param_mult",
            "hidden_lr",
            "hidden_grad",
            "unembed_init_var",
            "unembed_param_mult",
            "unembed_lr",
            "unembed_grad",
        ],
    )

    base_params = {
        "sp": Parameterization(
            embed_init_var=0.0,
            embed_param_mult=0.0,
            embed_lr=0.0,
            embed_grad=0.5,
            hidden_init_var=1.0,
            hidden_param_mult=0.0,
            hidden_lr=1.0,
            hidden_grad=0.5,
            unembed_init_var=1.0,
            unembed_param_mult=0.0,
            unembed_lr=1.0,
            unembed_grad=0.0,
        ),
        "mup": Parameterization(
            embed_init_var=1.0,
            embed_param_mult=0.5,
            embed_lr=0.5,
            embed_grad=0.5,
            hidden_init_var=1.0,
            hidden_param_mult=0.0,
            hidden_lr=1.0,
            hidden_grad=1.0,
            unembed_init_var=1.0,
            unembed_param_mult=0.5,
            unembed_lr=0.5,
            unembed_grad=0.5,
        ),
        "ntk": Parameterization(
            embed_init_var=0.0,
            embed_param_mult=0.0,
            embed_lr=0.0,
            embed_grad=0.5,
            hidden_init_var=0.0,
            hidden_param_mult=0.5,
            hidden_lr=0.5,
            hidden_grad=1.0,
            unembed_init_var=0.0,
            unembed_param_mult=0.5,
            unembed_lr=0.5,
            unembed_grad=0.5,
        ),
        "mean-field": Parameterization(
            embed_init_var=0.0,
            embed_param_mult=0.0,
            embed_lr=0.0,
            embed_grad=1.0,
            hidden_init_var=0.0,
            hidden_param_mult=0.5,
            hidden_lr=0.5,
            hidden_grad=1.5,
            unembed_init_var=0.0,
            unembed_param_mult=1.0,
            unembed_lr=0.0,
            unembed_grad=1.0,
        ),
    }

    style = style.lower()
    if style not in base_params:
        raise ValueError(f"Unknown parameterization style: {style}")

    params = base_params[style]._asdict()

    if not fully_aligned:
        if style == "sp":
            params.update(
                {
                    "embed_lr": 0.0,
                    "hidden_lr": 0.5,
                    "unembed_lr": 0.5,
                }
            )
        elif style == "mup":
            params.update(
                {
                    "embed_lr": 0.5,
                    "hidden_lr": 0.5,
                    "unembed_lr": 0.0,
                }
            )
        elif style == "ntk":
            params.update(
                {
                    "embed_lr": 0.0,
                    "hidden_lr": 0.0,
                    "unembed_lr": 0.0,
                }
            )
        elif style == "mean-field":
            params.update(
                {
                    "embed_lr": 0.0,
                    "hidden_lr": 0.0,
                    "unembed_lr": -0.5,
                }
            )

    return Parameterization(**params)


@pytree_dataclass
class SyntheticMetrics:
    avg_confidence: f32[b""]
    avg_char_confidence: f32[b""]
    max_char_confidence: f32[b""]
    avg_start_char_confidence: f32[b""]
    avg_final_char_confidence: f32[b""]


@pytree_dataclass
class Model:
    kernel: f32["layers groups d_hidden in_channels kernel_size"]
    linear: f32["layers d_hidden d_model"]
    ln: f32["layers d_model/t/d"]
    embed: f32["vocab/t d_model/d"]
    unembed: f32["vocab/t d_model/d"]

    @staticmethod
    @typechecked
    def init(h: Hparams, rng: PRNGKey) -> "Model":
        assert h.d_model % h.groups == 0
        in_channels, out_channels = (
            h.d_model // h.groups,
            h.d_hidden // h.groups,
        )
        truncated_normal_stddev = 0.87962566103423978

        embed_scale = 1.0 / (math.sqrt(h.d_model) * truncated_normal_stddev)
        embed = embed_scale * jax.random.normal(
            jax_extra.fold_in_str(rng, "embed"),
            (h.vocab, h.d_model),
            dtype=jnp.float32,
        )
        unembed_scale = 1.0 / (math.sqrt(h.d_model) * truncated_normal_stddev)
        unembed = unembed_scale * jax.random.truncated_normal(
            jax_extra.fold_in_str(rng, "unembed"),
            -2,
            2,
            (h.vocab, h.d_model),
            dtype=jnp.float32,
        )
        # TODO: figure out how to initialize kernel scaling
        kernel_scale = 1.0
        kernel_shape = (
            h.layers,
            h.groups,
            out_channels,
            in_channels,
            h.kernel_size,
        )

        kernel = kernel_scale * jax.random.normal(
            rng,
            kernel_shape,
        )
        linear_shape = (h.layers, h.d_hidden, h.d_model)
        linear_scale = 1.0 / (math.sqrt(h.d_hidden) * truncated_normal_stddev)
        linear = linear_scale * jax.random.normal(
            rng,
            linear_shape,
        )
        ln = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        arrays = Model(
            kernel=kernel, linear=linear, ln=ln, embed=embed, unembed=unembed
        )
        shardings = make_shardings(Model)
        return jax.tree.map(lax.with_sharding_constraint, arrays, shardings)

    def forward_pass(
        self, h: Hparams, ids: u32[b"B/d L"], is_seq_start: bool_[b"batch/d len"]
    ) -> f32[b"B/d L M/t"]:
        embed = shardops.all_gather("V/t M/d -> V/t M", jnp.bfloat16(self.embed))
        one_hot_ids = jax.nn.one_hot(ids, self.embed.shape[0])
        x = shardops.einsum_unreduced("B/d L V/t, V/t M -> B/d L M", one_hot_ids, embed)
        x = einops.rearrange(x, "B L (g fan_in) -> g fan_in B L", g=h.groups)

        segment_ids = jnp.cumsum(is_seq_start, axis=1)
        # TODO: mask is not used, assess how we work it in if needed
        segment_mask: bool_[b"B/d L L"] = (
            segment_ids[:, :, jnp.newaxis] == segment_ids[:, jnp.newaxis, :]
        )

        @explicit_activation_checkpointing
        @typechecked
        def loop_body(
            x: bf16[b"g fan_in B/d L/t"], layer_weights: Any
        ) -> Tuple[bf16[b"g fan_in B/d L/t"], Tuple[()]]:
            kernel, linear, ln = layer_weights

            out = jax.vmap(
                lambda x, k: jax.lax.conv_general_dilated(
                    x,
                    k,
                    window_strides=(1,),
                    padding="SAME",
                    dimension_numbers=("CNH", "OIH", "NCH"),
                )
            )(x, kernel)

            out = jax.nn.relu(out)
            out = layer_norm(out) * ln
            out = jax.lax.dropout(out, rate=h.dropout)
            out = shardops.einsum_unreduced(
                "g fan_out B/d L/t, fan_out fan_in -> g fan_in B/d L/t", out, linear
            )

            return jnp.bfloat16(x + out), ()

        x, () = jax.lax.scan(
            loop_body,
            jnp.bfloat16(x),
            (self.kernel, self.linear, self.ln),
        )
        unembed = shardops.all_gather("V/t M/d -> V/t M", jnp.bfloat16(self.unembed))
        logits = shardops.einsum_unreduced(
            "B/d L M, V/t M -> B/d L V/t",
            x,
            unembed,
            preferred_element_type=jnp.float32,
        )
        return logits

    @typechecked
    def loss(self, h: Hparams, batch: TokenBatch) -> Tuple[f32[b""], SyntheticMetrics]:
        # Given sequence-packed targets:
        #   [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
        # we want inputs:
        #   [[0, 1], [0, 3, 4], [0, 6, 7, 8]]
        # which we get by shifting the targets right by 1 and
        # masking sequence-start tokens to 0.
        inputs = jnp.pad(batch.targets[:, :-1], pad_width=((0, 0), (1, 0)))
        is_seq_start: bool_[b"batch/d len"] = batch.is_seq_start
        inputs: u32[b"batch/d len"] = jnp.where(is_seq_start, 0, inputs)

        logits: f32[b"batch/d len V/t"] = self.forward_pass(h, inputs, is_seq_start)
        max_logits: f32[b"batch/d len 1"] = lax.pmax(
            jnp.max(lax.stop_gradient(logits), axis=-1, keepdims=True), "t"
        )
        logits = logits - max_logits
        sum_logits = lax.psum(jnp.sum(jnp.exp(logits), axis=-1, keepdims=True), "t")
        logsumexp = jnp.log(sum_logits)
        logprobs: f32[b"batch/d len V/t"] = logits - logsumexp
        logprobs_at_targets = shardops.index_unreduced(
            "batch/d len [V/t], batch/d len -> batch/d len", logprobs, batch.targets
        )
        logprobs_at_targets = shardops.psum_scatter(
            "batch/d len -> batch/d len/t", logprobs_at_targets
        )
        if batch.loss_masks is not None:
            logprobs_at_targets = jnp.where(batch.loss_masks, logprobs_at_targets, 0)
        tokens_in_global_batch = logprobs_at_targets.size * jax.lax.psum(1, ("d", "t"))

        probs_at_targets = jnp.exp(logprobs_at_targets)

        batch_size, length = probs_at_targets.shape

        if batch.comment_starts is not None and batch.comment_ends is not None:
            comment_starts: u32[b"batch/d n_print"] = batch.comment_starts
            comment_ends: u32[b"batch/d n_print"] = batch.comment_ends

            batch_indices = jnp.arange(batch_size)[:, jnp.newaxis]  # (batch, 1)
            start_char_probs = probs_at_targets[batch_indices, comment_starts]
            avg_start_char_probs: f32[b""] = jnp.mean(start_char_probs)
            last_char_probs = probs_at_targets[batch_indices, comment_ends - 1]
            avg_last_char_probs: f32[b""] = jnp.mean(last_char_probs)

            comment_mask = jax.vmap(
                lambda starts_row, ends_row: jax.vmap(
                    lambda start, end: (jnp.arange(length) >= start)
                    & (jnp.arange(length) < end)
                )(starts_row, ends_row)
            )(comment_starts, comment_ends)

            probs_at_targets = probs_at_targets[:, jnp.newaxis, :]

            p_answer = jnp.prod(jnp.where(comment_mask, probs_at_targets, 1), axis=-1)

            # average confidence for each prints in sequence
            avg_p_answer: f32[b""] = jnp.mean(p_answer)

            total_tokens = jnp.sum(comment_ends - comment_starts + 1)
            comment_probs = jnp.where(comment_mask, probs_at_targets, 0)
            average_char_confidence = jnp.sum(comment_probs) / total_tokens
            max_char_confidence = jnp.max(comment_probs)

            synth_metrics = SyntheticMetrics(
                avg_confidence=avg_p_answer,
                max_char_confidence=max_char_confidence,
                avg_char_confidence=average_char_confidence,
                avg_start_char_confidence=avg_start_char_probs,
                avg_final_char_confidence=avg_last_char_probs,
            )
        else:
            synth_metrics = SyntheticMetrics(
                avg_confidence=jnp.float32(0.0),
                max_char_confidence=jnp.float32(0.0),
                avg_char_confidence=jnp.float32(0.0),
                avg_start_char_confidence=jnp.float32(0.0),
                avg_final_char_confidence=jnp.float32(0.0),
            )

        return (
            -jnp.sum(logprobs_at_targets) / jnp.float32(tokens_in_global_batch),
            synth_metrics,
        )


@pytree_dataclass
class RopeTable:
    sin: f32["len d_head2"]
    cos: f32["len d_head2"]

    @staticmethod
    def create(max_len: int, hparams: Hparams) -> "RopeTable":
        rope_max_timescale = hparams.rope_max_timescale
        d_head = hparams.d_head
        d = d_head // 2
        # endpoint=False is equivalent to what MaxText does. endpoint=True would be more natural, though.
        timescale = jnp.logspace(
            0, jnp.log10(jnp.float32(rope_max_timescale)), d, endpoint=False
        )
        position = jnp.arange(max_len, dtype=jnp.int32)
        sinusoid_inp = jnp.float32(position[:, jnp.newaxis]) / timescale[jnp.newaxis, :]
        sin = jnp.sin(sinusoid_inp)
        cos = jnp.cos(sinusoid_inp)
        return RopeTable(sin=sin, cos=cos)

    def apply(self, rearrange_spec, x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        sin = einops.rearrange(self.sin, rearrange_spec)
        cos = einops.rearrange(self.cos, rearrange_spec)
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
        return jnp.append(r1, r2, axis=-1)


@typechecked
def rms_norm(x: bf16[b"batch/d len M"]) -> bf16[b"batch/d len M"]:
    mean2 = save_for_backward(
        jnp.mean(jax.lax.square(jnp.float32(x)), axis=-1, keepdims=True)
    )
    return jnp.bfloat16(x * jax.lax.rsqrt(mean2 + 1e-6))


def layer_norm(x: bf16[b"batch/d len M"]) -> bf16[b"batch/d len M"]:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(variance + 1e-6)


@pytree_dataclass
class Metrics:
    loss: f32[b""]
    learning_rate: f32[b""]
    grad_norm: f32[b""]
    raw_grad_norm: f32[b""]


@dataclass(frozen=True)
class TrainingHparams:
    adam_b1: float
    adam_b2: float
    adam_eps: float
    adam_eps_root: float
    weight_decay: float
    warmup_steps: int
    steps: int
    steps_for_lr: int
    cosine_learning_rate_final_fraction: float
    learning_rate: float
    tokens: TokenBatchParams
    seed: int
    queue: Optional[str] = None
    use_grad_clip: bool = True
    use_gpu: bool = False
    use_checkpoint: bool = False


@pytree_dataclass
class State:
    weights: Model
    adam_mu: Model
    adam_nu: Model

    @staticmethod
    def init(hparams: Hparams, rng: PRNGKey) -> "State":
        weights = Model.init(hparams, rng)
        adam_mu = jax.tree.map(lambda p: p * 0.0, weights)
        adam_nu = jax.tree.map(lambda p: p * 0.0, weights)
        return State(weights=weights, adam_mu=adam_mu, adam_nu=adam_nu)


@partial(jax.jit, static_argnums=(2, 3), donate_argnums=(0,))
@shardtypes.scope
def training_step(
    state: State,
    step: u32[b""],
    h: Hparams,
    hparams: TrainingHparams,
    batch: TokenBatch,
) -> Tuple[Any, Metrics, SyntheticMetrics]:
    @partial(
        shardtypes.typed_shard_map, check_rep=False
    )  # check_rep=False for https://github.com/google/jax/issues/20335
    def sharded_step(
        state: State, step: u32[b""], batch: TokenBatch
    ) -> Tuple[State, Metrics, SyntheticMetrics]:
        (loss, synth_metrics), grad = jax.value_and_grad(
            lambda weights: weights.loss(h, batch), has_aux=True
        )(state.weights)
        # Gradients have already been reduced across chips because the gradient of the weight `all_gather`
        # is weight-gradient `psum_scatter`. Loss, on the other hand, hasn't been reduced across chips: if we
        # did that inside the autodiff, we'd be double-reducing the loss, effectively multiplying it by the
        # amount of data parallelism.
        #
        # So we reduce the loss across chips _outside_ the autodiff.
        loss = jax.lax.psum(loss, ("d", "t"))

        # Other than global-norm of gradients, no other communication is needed during the weight update,
        # because weights and grads are already fully sharded, as checked below.

        # Calculate learning rate from step number.
        # We use linear warmup then cosine decay. See https://arxiv.org/pdf/2307.09288.pdf section 2.2
        warmup_lr = (
            jnp.float32(step) / jnp.float32(hparams.warmup_steps)
        ) * hparams.learning_rate
        cosine = jnp.cos(
            jnp.pi
            * (
                jnp.float32(step - hparams.warmup_steps)
                / jnp.float32(hparams.steps_for_lr - hparams.warmup_steps)
            )
        )
        cosine_lr = hparams.learning_rate * (
            hparams.cosine_learning_rate_final_fraction
            + (1 - hparams.cosine_learning_rate_final_fraction) * (cosine * 0.5 + 0.5)
        )
        lr = jnp.where(step < hparams.warmup_steps, warmup_lr, cosine_lr)

        # AdamW optimizer with global gradient clipping.
        grad_leaves, grad_treedef = jax.tree_util.tree_flatten(grad)
        global_norm_square = jnp.float32(0.0)
        for g in grad_leaves:
            assert g.dtype == jnp.float32
            global_norm_square += jnp.sum(jax.lax.square(g))
        global_norm_square = jax.lax.psum(global_norm_square, ("d", "t"))
        global_norm = jnp.sqrt(global_norm_square)

        lr_scales = Model(
            embed=1.0,
            unembed=1.0,
            ln=1.0,
            kernel=1.0,
            linear=1.0,
        )

        if hparams.use_grad_clip:
            clip_value = 1.0
            rescale = jnp.minimum(1.0, clip_value / global_norm)
        else:
            rescale = 1.0

        new_ps = []
        new_mus = []
        new_nus = []
        for p, g, mu, nu, spec, lr_scale in zip(
            tree_leaves(state.weights),
            grad_leaves,
            tree_leaves(state.adam_mu),
            tree_leaves(state.adam_nu),
            tree_leaves(shardtypes.make_partition_specs(State)),
            tree_leaves(lr_scales),
        ):
            assert shardtypes.is_fully_sharded(
                spec
            ), "Weight update is only correctly scaled for fully sharded weights."
            # Gradient clipping
            g = g * rescale
            # Adam scaling
            mu = (1 - hparams.adam_b1) * g + hparams.adam_b1 * mu
            nu = (1 - hparams.adam_b2) * jax.lax.square(g) + hparams.adam_b2 * nu
            # We need step numbers to start at 1, not 0. Otherwise the bias correction produces NaN.
            completed_steps = step + 1
            mu_hat = mu / (1 - jnp.float32(hparams.adam_b1) ** completed_steps)
            nu_hat = nu / (1 - jnp.float32(hparams.adam_b2) ** completed_steps)
            # as per C.5. in https://arxiv.org/pdf2407.05872
            # they mention introducing hp a, b to below function,
            # TODO: test and see if a = b = something besides 1
            g = jnp.arctan2(mu_hat, jnp.sqrt(nu_hat))

            # Weight decay
            g += hparams.weight_decay * p
            # Learning rate
            g *= lr * lr_scale

            # Apply update
            new_ps.append(p - g)
            new_mus.append(mu)
            new_nus.append(nu)

        new_state = State(
            weights=jax.tree_util.tree_unflatten(grad_treedef, new_ps),
            adam_mu=jax.tree_util.tree_unflatten(grad_treedef, new_mus),
            adam_nu=jax.tree_util.tree_unflatten(grad_treedef, new_nus),
        )
        metrics = Metrics(
            loss=loss,
            learning_rate=lr,
            grad_norm=global_norm * rescale,
            raw_grad_norm=global_norm,
        )
        return new_state, metrics, synth_metrics

    return sharded_step(state, step, batch)


@dataclass(frozen=True)
class Paths:
    root_working_dir: str
    model_name: Optional[str]


@dataclass(frozen=True)
class MeshConfig:
    d: int
    t: int


@dataclass(frozen=True)
class Config:
    model: Hparams
    training: TrainingHparams
    paths: Paths
    num_hosts: int
    checkpoint_interval: int
    mesh: MeshConfig
    io: training_io.IOConfig
    flat_tokens: Optional[FlatTokensParams] = None
    hf_dataset: Optional[HuggingFaceDataParams] = None
    synthetic_dataset: Optional[SyntheticDataParams] = None

    def __post_init__(self):
        assert (
            self.flat_tokens is not None
            or self.hf_dataset is not None
            or self.synthetic_dataset is not None
        ), "Must provide either flat_tokens or hf_dataset or synthetic_dataset."
        assert not (
            self.flat_tokens is not None
            and self.hf_dataset is not None
            and self.synthetic_dataset is not None
        ), "Should not specify both flat_tokens and hf_dataset and synthetic_dataset."

    @cached_property
    def training_data(
        self,
    ) -> Union[FlatTokensParams, HuggingFaceDataParams, SyntheticDataParams]:
        return self.flat_tokens or self.hf_dataset or self.synthetic_dataset


def main_contained(config, logger):
    """Main program, which does not access external services except as specified by config.paths or logger."""
    # Use partitionable (and hopefully fusable!) RNG.
    #
    # This is slower in compute time than 'unsafe_rbg' with flag '--xla_tpu_spmd_rng_bit_generator_unsafe=true',
    # but hopefully faster in memory time because it's fusable.
    # TODO: check this is true and if not, provide our own that actually is fusable.
    jax.config.update("jax_threefry_partitionable", True)
    with Mesh(
        mesh_utils.create_device_mesh([config.mesh.d, config.mesh.t], jax.devices()),
        ("d", "t"),
    ):
        root_rng = jax.random.PRNGKey(config.training.seed)

        loader = get_loader("train", config.training_data, config.training.tokens)
        assert (
            config.model.vocab > loader.max_token_id
        ), f"{config.model.vocab} vs {loader.max_token_id}"
        config_name = hydra.core.hydra_config.HydraConfig.get()["job"]["config_name"]
        model_name = (
            config.paths.model_name
            if config.paths.model_name
            else get_model_name(config_name)
        )
        model_dir = os.path.join(config.paths.root_working_dir, model_name)
        print(model_name)

        state = jax.jit(partial(State.init, config.model))(
            fold_in_str(root_rng, "init")
        )
        if config.training.use_checkpoint:
            training_io.mkdir(model_dir)

        state, start_step = training_io.load_checkpoint_if_it_exists(
            model_dir, state, config.io
        )

        # Explicitly compile training step, to record XLA HLO graph.
        # See https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev
        c_training_step = training_step.lower(
            state, jnp.uint32(0), config.model, config.training, loader.load(0)
        ).compile()
        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # training_io.save_hlo_svg(os.path.join(model_dir, f'training_step_optimized_hlo_{date}.svg'), c_training_step)

        log_interval = math.ceil(config.training.steps / 5000)
        print(f"{log_interval=}")

        cum_metrics = None

        def update_metrics(metrics: Metrics):
            nonlocal cum_metrics
            cum_metrics.loss += metrics.loss
            cum_metrics.grad_norm += metrics.grad_norm
            cum_metrics.raw_grad_norm += metrics.raw_grad_norm
            cum_metrics.learning_rate += metrics.learning_rate

        for step in range(start_step, config.training.steps):
            if (
                config.training.use_checkpoint
                and step % config.checkpoint_interval == 0
                and step > start_step
            ):
                training_io.save_checkpoint(model_dir, step, state, config.io)

            # We profile on the second step, because the first step has a long pause for XLA
            # compilation and initial shuffle buffer loading.
            if training_io.is_device_0() and step == start_step + 1:
                jax.block_until_ready(state)
                training_io.start_profile()
                profile_start = time.time()

            # if half way point, double seq length and halve batch size
            if step == config.training.steps // 2:
                print("updating seq length and batch size")
                tokens = dataclasses.replace(
                    config.training.tokens,
                    len=config.training.tokens.len * 2,
                    batch=max(config.mesh.d, config.training.tokens.batch // 2),
                )
                config = dataclasses.replace(
                    config, training=dataclasses.replace(config.training, tokens=tokens)
                )
                loader = get_loader(
                    "train", config.training_data, config.training.tokens
                )
                c_training_step = training_step.lower(
                    state,
                    jnp.uint32(0),
                    config.model,
                    config.training,
                    loader.load(step),
                ).compile()

            batch = loader.load(step)
            state, output, synth_metrics = c_training_step(
                state, jnp.uint32(step), batch
            )

            # Run profile for two steps, to include data loading time in between them.
            if training_io.is_device_0() and step == start_step + 2:
                jax.block_until_ready(state)
                profile_duration = time.time() - profile_start
                training_io.stop_profile(model_dir)

                # Print MFU, including (one step of) data loading time.
                print(f"Profile time: {profile_duration}s for 2 steps.")
                model_params = jax.tree.reduce(
                    operator.add, jax.tree.map(lambda w: w.size, state.weights)
                )
                tokens = config.training.tokens.batch * config.training.tokens.len
                print(f"Model params: {model_params:_}")
                print(f"Tokens: {tokens:_}")
                device_flops = training_io.get_flops_per_device()
                num_devices = jax.device_count()
                print(
                    f"MFU (projections only): {100 * (2 * 6 * model_params * tokens / (num_devices * profile_duration)) / device_flops:.2f}% MFU"
                )

            if step % log_interval == 0:
                if cum_metrics:
                    cum_metrics = Metrics(
                        loss=cum_metrics.loss / log_interval,
                        learning_rate=cum_metrics.learning_rate / log_interval,
                        grad_norm=cum_metrics.grad_norm / log_interval,
                        raw_grad_norm=cum_metrics.raw_grad_norm / log_interval,
                    )
                else:
                    cum_metrics = output
                if batch.loss_masks is not None:
                    training_io.log(step, logger, synth_metrics)
                training_io.log(step, logger, cum_metrics)
                cum_metrics = output
            else:
                update_metrics(output)


def clear_tpu_locks():
    try:
        raw_pids = subprocess.run(
            ["lsof", "-w", "/dev/accel0"], capture_output=True, text=True
        ).stdout
        pids = set()
        for line in raw_pids.splitlines()[1:]:
            parts = line.split()
            if len(parts) > 1:
                pids.add(parts[1])
        for pid in pids:
            os.kill(int(pid), signal.SIGTERM)
        if pids:
            os.remove("/tmp/libtpu_lockfile")
    except Exception as e:
        print(f"Error clearing TPU locks: {e}")
        pass


def get_model_name(config_name: str):
    overrides = hydra.core.hydra_config.HydraConfig.get()["job"]["override_dirname"]
    ignore_overrides = [
        "training.queue",
    ]
    overrides = [
        override.lstrip("+")
        for override in overrides.split(",")
        if override.lstrip("+").split("=")[0] not in ignore_overrides
    ]

    overrides = "_".join(overrides)
    return f"{config_name}_{overrides}" if overrides else config_name


@hydra.main(config_path="configs", version_base=None)
def main(config):
    config = jax_extra.make_dataclass_from_dict(Config, config)
    if config.training.queue:
        config_name = hydra.core.hydra_config.HydraConfig.get()["job"]["config_name"]
        task_name = (
            config.paths.model_name
            if config.paths.model_name
            else get_model_name(config_name)
        )
        git_branch_name = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        task = Task.init(
            project_name=f"{config_name}/{git_branch_name}", task_name=task_name
        )

        if config.training.use_gpu:
            task.set_packages("requirements-gpu.txt")
        else:
            task.set_packages("requirements-tpu.txt")

        task.add_tags([git_branch_name])
        logger = task.get_logger()
        task.execute_remotely(queue_name=config.training.queue)
        task.launch_multi_node(
            config.num_hosts, wait=True, queue=config.training.queue + "-workers"
        )
        clear_tpu_locks()
        jax.distributed.initialize(
            os.environ["MASTER_ADDR"] + ":" + os.environ["MASTER_PORT"],
            num_processes=int(os.environ["WORLD_SIZE"]),
            process_id=int(os.environ["RANK"]),
        )
    else:
        logger = None
    main_contained(config, logger)

    if not training_io.is_device_0():
        task.set_system_tags((task.get_system_tags() or []) + ["hidden"])


if __name__ == "__main__":
    main()
