from typing import Any, Tuple, Dict, Union

import equinox as eqx
import jax.random as jrandom
import numpy as np

from jax_learning.agents import LearningAgent
from jax_learning.buffers import ReplayBuffer
from jax_learning.common import EpochSummary, load_checkpoint
from jax_learning.constants import (
    EXPLORATION_STRATEGY,
    CONTINUOUS,
    DISCRETE,
    LEARNER,
    OFFLINE,
    ONLINE,
    LEARNERS,
)
from jax_learning.learners import ReinforcementLearner, Learner

AGENT_KEY = "agent_key"
EPS = "eps"
EPS_WARMUP = "eps_warmup"
EPS_DECAY = "eps_decay"
MIN_EPS = "min_eps"


class RLAgent(LearningAgent):
    def __init__(
        self,
        model: Dict[str, eqx.Module],
        model_key: str,
        buffer: ReplayBuffer,
        learner: ReinforcementLearner,
        key: jrandom.PRNGKey,
    ):
        super().__init__(model, buffer, learner)
        self._model_key = model_key
        self._key = key

    @property
    def model_key(self):
        return self._model_key

    def deterministic_action(
        self, obs: np.ndarray, h_state: np.ndarray, info: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        obs = self._obs_rms.normalize(obs) if self._obs_rms else obs
        action, next_h_state = self.model[self._model_key].deterministic_action(
            obs, h_state
        )
        return np.asarray(action), np.asarray(next_h_state)

    def compute_action(
        self,
        obs: np.ndarray,
        h_state: np.ndarray,
        info: dict,
        overwrite_rng_key: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        obs = self._obs_rms.normalize(obs) if self._obs_rms else obs
        new_key, curr_key = jrandom.split(self._key)
        action, next_h_state = self.model[self._model_key].random_action(
            obs, h_state, curr_key
        )

        if overwrite_rng_key:
            self._key = new_key

        return np.asarray(action), np.asarray(next_h_state)

    def checkpoint(self) -> Dict[str, Any]:
        agent_dict = super().checkpoint()
        agent_dict[AGENT_KEY] = self._key
        return agent_dict

    def load(self, load_path: str):
        agent_dict = load_checkpoint(load_path)
        self._key = agent_dict[AGENT_KEY]
        self._learner.load(agent_dict[LEARNER])


class EpsilonGreedyAgent(RLAgent):
    def __init__(
        self,
        model: Dict[str, eqx.Module],
        model_key: str,
        buffer: ReplayBuffer,
        learner: ReinforcementLearner,
        init_eps: float,
        min_eps: float,
        eps_decay: float,
        eps_warmup: float,
        action_space: str,
        action_dim: int,
        key: jrandom.PRNGKey,
        action_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        assert action_space in (CONTINUOUS, DISCRETE)
        super().__init__(model, model_key, buffer, learner, key)
        self._eps = init_eps
        self._init_eps = init_eps
        self._min_eps = min_eps
        self._eps_decay = eps_decay
        self._eps_warmup = eps_warmup
        self._action_dim = action_dim
        self._action_range = action_range
        self._action_midpoint = (action_range[0] + action_range[1]) / 2
        self._action_scale = (action_range[1] - action_range[0]) / 2
        if action_space == CONTINUOUS:
            self.compute_action = self.compute_action_continuous
        elif action_space == DISCRETE:
            self.compute_action = self.compute_action_discrete
        else:
            raise NotImplementedError

    def compute_action_continuous(
        self,
        obs: np.ndarray,
        h_state: np.ndarray,
        info: dict,
        overwrite_rng_key: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        obs = self._obs_rms.normalize(obs) if self._obs_rms else obs
        new_key, curr_key = jrandom.split(self._key)
        # action, next_h_state = self.model[self._model_key].deterministic_action(
        #     obs, h_state
        # )
        action, next_h_state = self.model[self._model_key].random_action(
            obs, h_state, curr_key
        )
        if jrandom.bernoulli(key=curr_key, p=self._eps):
            sign = (-1) ** jrandom.randint(
                curr_key, shape=(self._action_dim,), minval=0, maxval=2
            )
            action = (
                sign
                * jrandom.uniform(
                    curr_key, shape=(self._action_dim,), minval=0, maxval=1
                )
                * self._action_scale
                + self._action_midpoint
            )
            info[EXPLORATION_STRATEGY] = 0
        else:
            info[EXPLORATION_STRATEGY] = 1

        if overwrite_rng_key:
            self._key = new_key
            if self._eps_warmup > 0:
                self._eps_warmup -= 1
            else:
                self._eps = max(self._eps * self._eps_decay, self._min_eps)

        return np.asarray(action), np.asarray(next_h_state)

    def compute_action_discrete(
        self,
        obs: np.ndarray,
        h_state: np.ndarray,
        info: dict,
        overwrite_rng_key: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        obs = self._obs_rms.normalize(obs) if self._obs_rms else obs
        new_key, curr_key = jrandom.split(self._key)
        action, next_h_state = self.model[self._model_key].deterministic_action(
            obs, h_state
        )
        if jrandom.bernoulli(key=curr_key, p=self._eps):
            action = jrandom.randint(
                curr_key, shape=(1,), minval=0, maxval=self._action_dim
            ).item()
            info[EXPLORATION_STRATEGY] = 0
        else:
            info[EXPLORATION_STRATEGY] = 1

        if overwrite_rng_key:
            self._key = new_key
            if self._eps_warmup > 0:
                self._eps_warmup -= 1
            else:
                self._eps = max(self._eps * self._eps_decay, self._min_eps)

        return np.asarray(action), np.asarray(next_h_state)

    def checkpoint(self) -> Dict[str, Any]:
        agent_dict = super().checkpoint()
        agent_dict[EPS] = self._eps
        agent_dict[EPS_DECAY] = self._eps_decay
        agent_dict[EPS_WARMUP] = self._eps_warmup
        agent_dict[MIN_EPS] = self._min_eps
        agent_dict[AGENT_KEY] = self._key
        return agent_dict

    def load(self, load_path: str):
        agent_dict = load_checkpoint(load_path)
        self._key = agent_dict[AGENT_KEY]
        self._eps = agent_dict[EPS]
        self._eps_decay = agent_dict[EPS_DECAY]
        self._eps_warmup = agent_dict[EPS_WARMUP]
        self._min_eps = agent_dict[MIN_EPS]
        self._learner.load(agent_dict[LEARNER])


class OfflineOnlineRLAgent(RLAgent):
    def __init__(
        self,
        model: Dict[str, eqx.Module],
        model_key: str,
        buffer: ReplayBuffer,
        learners: Dict[str, Union[Learner, ReinforcementLearner]],
        key: jrandom.PRNGKey,
    ):
        assert (
            len(learners) == 2
        ), "There should only be an offline and an online learning algorithm"
        super().__init__(model, model_key, buffer, learners[ONLINE], key)
        self._learners = learners

    @property
    def model_key(self):
        return self._model_key

    def pretrain(self, learn_info: dict, epoch_summary: EpochSummary, **kwargs):
        self._learners[OFFLINE].learn(learn_info, epoch_summary, **kwargs)

    def learn(self, learn_info: dict, epoch_summary: EpochSummary, **kwargs):
        self._learners[ONLINE].learn(learn_info, epoch_summary, **kwargs)

    def checkpoint(self) -> Dict[str, Any]:
        agent_dict = {
            AGENT_KEY: self._key,
            LEARNERS: {
                OFFLINE: self._learners[OFFLINE].checkpoint(),
                ONLINE: self._learners[ONLINE].checkpoint(),
            },
        }
        return agent_dict

    def load(self, load_path: str):
        agent_dict = load_checkpoint(load_path)
        self._key = agent_dict[AGENT_KEY]
        self._learners[OFFLINE].load(agent_dict[LEARNERS][OFFLINE])
        self._learners[ONLINE].load(agent_dict[LEARNERS][ONLINE])
