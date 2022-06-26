import jax.numpy as jnp


def get_lprob(dist, x):
    return jnp.sum(dist.lprob(x), keepdims=True)
