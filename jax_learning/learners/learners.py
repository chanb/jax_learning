from abc import abstractmethod
from argparse import Namespace
from typing import Dict

import equinox as eqx
import jax
import numpy as np
import optax

from jax_learning.buffers.buffers import ReplayBuffer


class Learner:
    def __init__(self,
                 model: Dict[str, eqx.Module],
                 opt: Dict[str, optax.GradientTransformation],
                 buffer: ReplayBuffer,
                 cfg: Namespace):
        self._model = model
        self._opt = opt
        self._opt_state = {model_key: model_opt.init(model[model_key]) for model_key, model_opt in self._opt.items()}
        self._buffer = buffer
        self._cfg = cfg

    @property
    def buffer(self):
        return self._buffer
    
    @property
    def opt_state(self):
        return self._opt_state

    @property
    def model(self):
        return self._model

    @property
    def opt(self):
        return self._opt

    @property
    def cfg(self):
        return self._cfg

    @abstractmethod
    def learn(self,
              next_obs: np.ndarray,
              next_h_state: np.ndarray,
              learn_info: dict):
        raise NotImplementedError


class LearnerWithTargetNetwork(Learner):
    def __init__(self,
                 model: Dict[str, eqx.Module],
                 target_model: Dict[str, eqx.Module],
                 opt: Dict[str, optax.GradientTransformation],
                 buffer: ReplayBuffer,
                 cfg: Namespace):
        super().__init__(model, opt, buffer, cfg)
        self._target_model = target_model
    
    @property
    def target_model(self):
        return self._target_model

    def polyak_average(self, model_key):
        self._target_model[model_key] = jax.tree_map(lambda p, tp: p * self._tau + tp * (1 - self._tau),
                                                     self.model[model_key],
                                                     self.target_model[model_key])
