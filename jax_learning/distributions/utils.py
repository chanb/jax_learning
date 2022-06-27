import jax.numpy as jnp
import numpy as np

from jax_learning.distributions import Distribution


def get_lprob(dist: Distribution, x: np.ndarray) -> np.ndarray:
    return jnp.sum(dist.lprob(x), keepdims=True)
