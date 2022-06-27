from abc import abstractmethod
from argparse import Namespace
from typing import Dict

import equinox as eqx
import jax
import numpy as np
import optax

from jax_learning.buffers import ReplayBuffer
from jax_learning.common import RunningMeanStd, polyak_average_generator
from jax_learning.constants import NORMALIZE_OBS, NORMALIZE_VALUE


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

        self._obs_rms = False
        self._val_rms = False
        if getattr(self._cfg, NORMALIZE_OBS, False):
            self._obs_rms = RunningMeanStd(shape=cfg.obs_dim)
        if getattr(self._cfg, NORMALIZE_VALUE, False):
            self._val_rms = RunningMeanStd(shape=cfg.rew_dim)

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

    @property
    def obs_rms(self):
        return self._obs_rms

    @property
    def val_rms(self):
        return self._val_rms

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
        self._tau = cfg.tau
        self.polyak_average = polyak_average_generator(self._tau)
    
    @property
    def target_model(self):
        return self._target_model

    def update_target_model(self, model_key):
        self._target_model[model_key] = jax.tree_map(self.polyak_average,
                                                     self.model[model_key],
                                                     self.target_model[model_key])
