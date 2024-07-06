import os
import operator
import math
import time
import datetime
from functools import partial
from typing import List
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax_extra import fold_in_str, make_dataclass_from_dict
import hydra
from omegaconf.listconfig import ListConfig

import training_io
from input_loader import (
    get_loader,
)
from train import State, training_step, Config as ModelConfig


def train_model(config, steps):
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
        # model_dir = os.path.join(config.paths.root_working_dir, model_name)

        # training_io.mkdir(model_dir)
        state = jax.jit(partial(State.init, config.model))(
            fold_in_str(root_rng, "init")
        )
        # state, start_step = training_io.load_checkpoint_if_it_exists(model_dir, state, config.io)

        # Explicitly compile training step, to record XLA HLO graph.
        # See https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev
        c_training_step = training_step.lower(
            state, jnp.uint32(0), config.model, config.training, loader.load(0)
        ).compile()
        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # training_io.save_hlo_svg(os.path.join(model_dir, f'training_step_optimized_hlo_{date}.svg'), c_training_step)

        for step in range(steps):
            state, output = c_training_step(state, jnp.uint32(step), loader.load(step))
            print("completed step")
            training_io.log(step, None, output)
        print(f"{state.weights.coord_checks_per_activation}")


@dataclass(frozen=True)
class CoordChecksConfig:
    base_model: str
    model_family: ListConfig 
    steps: int

@hydra.main(config_path="configs", version_base=None)
def main(config ):
    print(f"{type(config)}")
    config = make_dataclass_from_dict(CoordChecksConfig, config)

    for config_name in config.model_family:
        cfg = hydra.compose(config_name=config_name)
        model_config = make_dataclass_from_dict(ModelConfig, cfg)
        print(f'training {config_name}')
        train_model(model_config, config.steps) 
if __name__ == "__main__":
    main()