import jax
import jax.numpy as jnp
from shardlib.shardtypes import f32, u32, bf16, pytree_dataclass
from typing import Any
from dataclasses import dataclass


PRNGKey = Any


@dataclass(frozen=True)
class Hparams:
    d_model: int


@pytree_dataclass
class Model:
    def init(h: Hparams, rng: PRNGKey) -> "Model":
        pass
