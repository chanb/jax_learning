from abc import abstractmethod
from argparse import Namespace
from turtle import update
from typing import Any, Dict

import equinox as eqx
import jax
import numpy as np
import optax

from jax_learning.buffers import ReplayBuffer
from jax_learning.common import EpochSummary, RunningMeanStd, polyak_average_generator
from jax_learning.constants import NORMALIZE_OBS, NORMALIZE_VALUE, STEP

MODEL = "model"
OPT = "opt"
OPT_STATE = "opt_state"
OBS_RMS = "obs_rms"
VAL_RMS = "val_rms"
TARGET_MODEL = "target_model"


class Learner:
    def __init__(
        self,
        model: Dict[str, eqx.Module],
        opt: Dict[str, optax.GradientTransformation],
        buffer: ReplayBuffer,
        cfg: Namespace,
    ):
        self._step = 0
        self._model = model
        self._opt = opt
        self._opt_state = {
            model_key: model_opt.init(model[model_key])
            for model_key, model_opt in self._opt.items()
        }
        self._buffer = buffer
        self._cfg = cfg
        self._obs_rms = False
        if getattr(self._cfg, NORMALIZE_OBS, False):
            self._obs_rms = RunningMeanStd(shape=cfg.obs_dim)

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

    @abstractmethod
    def learn(
        self,
        learn_info: dict,
        epoch_summary: EpochSummary,
        **kwargs,
    ):
        raise NotImplementedError

    def checkpoint(
        self,
    ) -> Dict[str, Any]:
        return {
            MODEL: self.model,
            OPT_STATE: self.opt_state,
            OBS_RMS: self.obs_rms,
            STEP: self._step,
        }

    def load(self, data: Dict[str, Any]):
        self._obs_rms = data[OBS_RMS]
        self._opt_state = data[OPT_STATE]
        self._step = data[STEP]
        for model_key, model_filename in data[MODEL].items():
            self.model[model_key] = eqx.tree_deserialise_leaves(
                model_filename, self.model[model_key]
            )


class ReinforcementLearner(Learner):
    def __init__(
        self,
        model: Dict[str, eqx.Module],
        opt: Dict[str, optax.GradientTransformation],
        buffer: ReplayBuffer,
        cfg: Namespace,
    ):
        super().__init__(model, opt, buffer, cfg)
        self._update_frequency = cfg.update_frequency
        self._gamma = cfg.gamma
        self._val_rms = False
        if getattr(self._cfg, NORMALIZE_VALUE, False):
            self._val_rms = RunningMeanStd(shape=cfg.rew_dim)

    @property
    def val_rms(self):
        return self._val_rms

    @property
    def update_frequency(self):
        return self._update_frequency

    @update_frequency.setter
    def update_frequency(self, update_frequency: int):
        self._update_frequency = update_frequency

    def checkpoint(
        self,
    ) -> Dict[str, Any]:
        learner_dict = super().checkpoint()
        learner_dict[VAL_RMS] = self.val_rms

    def load(self, data: Dict[str, Any]):
        super().load(data)
        self._val_rms = data[VAL_RMS]

    def checkpoint(
        self,
    ) -> Dict[str, Any]:
        return {
            MODEL: self.model,
            OPT_STATE: self.opt_state,
            OBS_RMS: self.obs_rms,
            VAL_RMS: self.val_rms,
        }

    def load(self, data: Dict[str, Any]):
        self._obs_rms = data[OBS_RMS]
        self._val_rms = data[VAL_RMS]
        self._opt_state = data[OPT_STATE]
        for model_key, model_filename in data[MODEL].items():
            self.model[model_key] = eqx.tree_deserialise_leaves(
                model_filename, self.model[model_key]
            )


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

    def load(self, data: Dict[str, Any]):
        super().load(data)
        for model_key, model_filename in data[TARGET_MODEL].items():
            self.target_model[model_key] = eqx.tree_deserialise_leaves(
                model_filename, self.target_model[model_key]
            )
