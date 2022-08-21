from abc import abstractstaticmethod
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import math
import numpy as np


class Distribution:
    @abstractstaticmethod
    def sample(
        *, key: jrandom.PRNGKey, num_samples: Optional[int] = None
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractstaticmethod
    def lprob(*, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Categorical(Distribution):
    @eqx.filter_jit
    @staticmethod
    def sample(
        logits: np.ndarray, key: jrandom.PRNGKey, num_samples: Optional[int] = None
    ) -> np.ndarray:
        if num_samples:
            shape = (num_samples, *logits.shape[-1])
            return jrandom.categorical(
                key=key, logits=logits, shape=shape, axis=-1
            ).astype(int)
        else:
            return jrandom.categorical(key=key, logits=logits, axis=-1).astype(int)

    @eqx.filter_jit
    @staticmethod
    def lprob(logits: np.ndarray, x: np.ndarray) -> np.ndarray:
        return logits[x.astype(int)] - jax.scipy.special.logsumexp(
            logits, axis=-1, keepdims=True
        )


class Normal(Distribution):
    @eqx.filter_jit
    @staticmethod
    def sample(
        mean: np.ndarray,
        std: np.ndarray,
        key: jrandom.PRNGKey,
        num_samples: Optional[int] = None,
    ) -> np.ndarray:
        if num_samples:
            shape = (num_samples, *mean.shape)
        else:
            shape = mean.shape
        return mean + jrandom.normal(key=key, shape=shape) * std

    @eqx.filter_jit
    def lprob(mean: np.ndarray, std: np.ndarray, x: np.ndarray) -> np.ndarray:
        var = std**2
        log_std = jnp.log(std)
        return (
            -((x - mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))
        )
