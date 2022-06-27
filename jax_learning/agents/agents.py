from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple

from jax_learning.buffers import ReplayBuffer
from jax_learning.constants import RESET
from jax_learning.learners import ReinforcementLearner

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

    def learn(self, next_obs: np.ndarray, next_h_state: np.ndarray, learn_info: dict):
        pass

    def store(
        self,
        obs: np.ndarray,
        h_state: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        info: dict,
        next_obs: np.ndarray,
        next_h_state: np.ndarray,
    ):
        self._buffer.push(obs, h_state, act, rew, done, info, next_obs, next_h_state)

    def reset(self):
        if hasattr(self.model, RESET):
            return self.model.reset()
        return np.array([0.0], dtype=np.float32)


class LearningAgent(Agent):
    def __init__(
        self, model: eqx.Module, buffer: ReplayBuffer, learner: ReinforcementLearner
    ):
        super().__init__(model, buffer)
        self._learner = learner
        self._obs_rms = learner.obs_rms

    @property
    def learner(self):
        return self._learner

    def store(
        self,
        obs: np.ndarray,
        h_state: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        info: dict,
        next_obs: np.ndarray,
        next_h_state: np.ndarray,
    ):
        if self._obs_rms:
            self._obs_rms.update(obs)
        self._buffer.push(obs, h_state, act, rew, done, info, next_obs, next_h_state)

    def learn(self, next_obs: np.ndarray, next_h_state: np.ndarray, learn_info: dict):
        self.learner.learn(next_obs, next_h_state, learn_info)
