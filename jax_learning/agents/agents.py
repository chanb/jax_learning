from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple, Any, Dict

from jax_learning.buffers import ReplayBuffer
from jax_learning.common import EpochSummary, load_checkpoint
from jax_learning.constants import RESET, LEARNER
from jax_learning.learners import Learner

import equinox as eqx
import numpy as np


class Agent(ABC):
    def __init__(self, model: eqx.Module, buffer: ReplayBuffer):
        self._model = model
        self._buffer = buffer

    @property
    def model(self):
        return self._model

    @property
    def buffer(self):
        return self._buffer

    @abstractproperty
    def learner(self):
        raise NotImplementedError

    @abstractmethod
    def compute_action(
        self, obs: np.ndarray, h_state: np.ndarray, act_info: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def deterministic_action(
        self, obs: np.ndarray, h_state: np.ndarray, act_info: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def pretrain(self, learn_info: dict, epoch_summary: EpochSummary, **kwargs):
        pass

    def learn(self, learn_info: dict, epoch_summary: EpochSummary, **kwargs):
        pass

    def store(
        self,
        obs: np.ndarray,
        h_state: np.ndarray,
        act: np.ndarray,
        rew: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        next_obs: np.ndarray,
        next_h_state: np.ndarray,
    ):
        self._buffer.push(
            obs, h_state, act, rew, terminated, truncated, info, next_obs, next_h_state
        )

    def reset(self):
        if hasattr(self.model, RESET):
            return self.model.reset()
        return np.array([0.0], dtype=np.float32)

    def checkpoint(self) -> Dict[str, Any]:
        return {}


class LearningAgent(Agent):
    def __init__(self, model: eqx.Module, buffer: ReplayBuffer, learner: Learner):
        super().__init__(model, buffer)
        self._learner = learner
        self._obs_rms = learner.obs_rms

    @property
    def learner(self):
        return self._learner

    @learner.setter
    def learner(self, learner):
        self._learner = learner

    def store(
        self,
        obs: np.ndarray,
        h_state: np.ndarray,
        act: np.ndarray,
        rew: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        next_obs: np.ndarray,
        next_h_state: np.ndarray,
    ):
        if self._obs_rms:
            self._obs_rms.update(obs)
        super().store(
            obs, h_state, act, rew, terminated, truncated, info, next_obs, next_h_state
        )

    def learn(self, learn_info: dict, epoch_summary: EpochSummary, **kwargs):
        self.learner.learn(learn_info, epoch_summary, **kwargs)

    def checkpoint(self) -> Dict[str, Any]:
        return {LEARNER: self._learner.checkpoint()}

    def load(self, load_path: str):
        agent_dict = load_checkpoint(load_path)
        self._learner.load(agent_dict[LEARNER])
