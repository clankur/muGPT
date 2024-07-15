# %%
import jax.numpy as jnp
w_kv = jnp.arange(8).reshape((2, 4))
jnp.where(
    jnp.arange(2)[None, :, None] < 1,
    w_kv * -1,
    w_kv * 5
)# %%
