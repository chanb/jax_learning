from abc import abstractmethod
from argparse import Namespace

import equinox as eqx
import numpy as np
import optax

from jax_learning.buffers.buffers import ReplayBuffer


class Learner:
    def __init__(self,
                 model: eqx.Module,
                 opt: optax.GradientTransformation,
                 buffer: ReplayBuffer,
                 cfg: Namespace):
        self._model = model
        self._opt = opt
        self._buffer = buffer
        self._cfg = cfg

    @property
    def model(self):
        return self._model

    @property
    def opt(self):
        return self._opt

    @property
    def buffer(self):
        return self._buffer

    @abstractmethod
    def learn(self,
              next_obs: np.ndarray,
              next_h_state: np.ndarray,
              learn_info: dict):
        raise NotImplementedError
