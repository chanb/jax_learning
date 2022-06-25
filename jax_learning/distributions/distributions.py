from abc import ABC, abstractmethod
from jax.core import NamedShape
from jax.scipy.stats.norm import logpdf as normal_lprob
from typing import Sequence, Union, Optional

import equinox as eqx
import jax
import jax.random as jrandom
import numpy as np


class Distribution(eqx.Module):
    @abstractmethod
    def sample(self,
               key: jrandom.PRNGKey,
               num_samples: Optional[int]=None) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def lprob(self,
              x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Categorical(Distribution):
    logits: np.ndarray

    def __init__(self,
                 logits: np.ndarray):
        self.logits = logits

    @eqx.filter_jit
    def sample(self,
               key: jrandom.PRNGKey,
               num_samples: Optional[int]=None) -> np.ndarray:
        if num_samples:
            shape = (num_samples, *self.logits.shape[-1])
        else:
            shape = self.logits.shape[-1]
        return jrandom.categorical(key=key, logits=self.logits, shape=shape, axis=-1)

    @eqx.filter_jit
    def lprob(self, x: np.ndarray) -> np.ndarray:
        return self.logits[x] - jax.scipy.special.logsumexp(self.logits, axis=-1, keepdims=True)


class Normal(Distribution):
    mean: np.ndarray
    std: np.ndarray

    def __init__(self,
                 mean: np.ndarray,
                 std: np.ndarray):
        self.mean = mean
        self.std = std

    @eqx.filter_jit
    def sample(self,
               key: jrandom.PRNGKey,
               num_samples: Optional[int]=None) -> np.ndarray:
        if num_samples:
            shape = (num_samples, *self.mean.shape)
        else:
            shape = self.mean.shape
        return self.mean + jrandom.normal(key=key, shape=shape) * self.std

    @eqx.filter_jit
    def lprob(self,
              x: np.ndarray) -> np.ndarray:
        lprob = normal_lprob(x, loc=self.mean, scale=self.std)
        return lprob
