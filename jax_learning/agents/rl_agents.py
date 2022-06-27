from typing import Tuple, Dict

import equinox as eqx
import jax.random as jrandom
import numpy as np

from jax_learning.agents import LearningAgent
from jax_learning.buffers import ReplayBuffer
from jax_learning.constants import EXPLORATION_STRATEGY
from jax_learning.learners import ReinforcementLearner


class RLAgent(LearningAgent):
    def __init__(self,
                 model: Dict[str, eqx.Module],
                 model_key: str,
                 buffer: ReplayBuffer,
                 learner: ReinforcementLearner,
                 key: jrandom.PRNGKey):
        super().__init__(model, buffer, learner)
        self._model_key = model_key
        self._key = key
        
    def deterministic_action(self,
                             obs: np.ndarray,
                             h_state: np.ndarray,
                             info: dict) -> Tuple[np.ndarray ,np.ndarray]:
        obs = self._obs_rms.normalize(obs) if self._obs_rms else obs
        action, next_h_state = self.model[self._model_key].deterministic_action(obs, h_state)
        return np.asarray(action), np.asarray(next_h_state)

    def compute_action(self,
                       obs: np.ndarray,
                       h_state: np.ndarray,
                       info: dict,
                       overwrite_rng_key: bool=True) -> Tuple[np.ndarray ,np.ndarray]:
        obs = self._obs_rms.normalize(obs) if self._obs_rms else obs
        new_key, curr_key = jrandom.split(self._key)
        action, next_h_state = self.model[self._model_key].random_action(obs, h_state, curr_key)

        if overwrite_rng_key:
            self._key = new_key

        return np.asarray(action), np.asarray(next_h_state)


class EpsilonGreedyAgent(RLAgent):
    def __init__(self,
                 model: Dict[str, eqx.Module],
                 model_key: str,
                 buffer: ReplayBuffer,
                 learner: ReinforcementLearner,
                 init_eps: float,
                 min_eps: float,
                 eps_decay: float,
                 eps_warmup: float,
                 key: jrandom.PRNGKey):
        super().__init__(model, model_key, buffer, learner, key)
        self._eps = init_eps
        self._init_eps = init_eps
        self._min_eps = min_eps
        self._eps_decay = eps_decay
        self._eps_warmup = eps_warmup
        
    def compute_action(self,
                       obs: np.ndarray,
                       h_state: np.ndarray,
                       info: dict,
                       overwrite_rng_key: bool=True) -> Tuple[np.ndarray ,np.ndarray]:
        obs = self._obs_rms.normalize(obs) if self._obs_rms else obs
        new_key, curr_key = jrandom.split(self._key)
        if jrandom.bernoulli(key=curr_key, p=self._eps):
            val, next_h_state = self.model[self._model_key].q_values(obs, h_state)
            action = jrandom.randint(curr_key, shape=(1,), minval=0, maxval=val.shape[-1]).item()
            info[EXPLORATION_STRATEGY] = 0
        else:
            action, next_h_state = self.model[self._model_key].deterministic_action(obs, h_state)
            info[EXPLORATION_STRATEGY] = 1

        if overwrite_rng_key:
            self._key = new_key
            if self._eps_warmup > 0:
                self._eps_warmup -= 1
            else:
                self._eps = max(self._eps * self._eps_decay, self._min_eps)

        return np.asarray(action), np.asarray(next_h_state)
