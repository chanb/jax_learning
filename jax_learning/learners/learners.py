from abc import abstractmethod
from argparse import Namespace
from typing import Any, Dict

import equinox as eqx
import jax
import numpy as np
import optax

from jax_learning.buffers import ReplayBuffer
from jax_learning.common import EpochSummary, RunningMeanStd, polyak_average_generator
from jax_learning.constants import NORMALIZE_OBS, NORMALIZE_VALUE

MODEL = "model"
OPT = "opt"
OPT_STATE = "opt_state"
OBS_RMS = "obs_rms"
VAL_RMS = "val_rms"
TARGET_MODEL = "target_model"


class ReinforcementLearner:
    def __init__(
        self,
        model: Dict[str, eqx.Module],
        opt: Dict[str, optax.GradientTransformation],
        buffer: ReplayBuffer,
        cfg: Namespace,
    ):
        self._model = model
        self._opt = opt
        self._opt_state = {
            model_key: model_opt.init(model[model_key])
            for model_key, model_opt in self._opt.items()
        }
        self._buffer = buffer
        self._cfg = cfg

        self._step = cfg.load_step
        self._gamma = cfg.gamma
        self._update_frequency = cfg.update_frequency

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
    def learn(
        self,
        next_obs: np.ndarray,
        next_h_state: np.ndarray,
        learn_info: dict,
        epoch_summary: EpochSummary,
    ):
        raise NotImplementedError

    def checkpoint(
        self,
    ) -> Dict[str, Any]:
        return {
            MODEL: self.model,
            OPT_STATE: self.opt_state,
            OBS_RMS: self.obs_rms,
            VAL_RMS: self.val_rms,
        }


class ReinforcementLearnerWithTargetNetwork(ReinforcementLearner):
    def __init__(
        self,
        model: Dict[str, eqx.Module],
        target_model: Dict[str, eqx.Module],
        opt: Dict[str, optax.GradientTransformation],
        buffer: ReplayBuffer,
        cfg: Namespace,
    ):
        super().__init__(model, opt, buffer, cfg)
        self._target_model = target_model
        self._target_update_frequency = cfg.target_update_frequency
        self._tau = cfg.tau
        self.polyak_average = polyak_average_generator(self._tau)

    @property
    def target_model(self):
        return self._target_model

    def update_target_model(self, model_key):
        self._target_model[model_key] = jax.tree_map(
            self.polyak_average, self.model[model_key], self.target_model[model_key]
        )

    def checkpoint(
        self,
    ) -> Dict[str, Any]:
        checkpoint_dict = super().checkpoint()
        checkpoint_dict[TARGET_MODEL] = self.target_model
        return checkpoint_dict
