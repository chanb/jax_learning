from abc import abstractmethod
from typing import Tuple, Sequence, Callable, Any, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


class Model(eqx.Module):
    @staticmethod
    def apply_function(func: Callable, *args: Sequence[Any], **kwargs: dict):
        return func(*args, **kwargs)


class Temperature(Model):
    log_temp: float

    def __init__(
        self,
        init_temp: float = 1.0,
    ):
        self.log_temp = jnp.log(init_temp)

    def __call__(self):
        return jnp.exp(self.log_temp)


class Policy(Model):
    @abstractmethod
    def deterministic_action(
        self, obs: np.ndarray, h_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class StochasticPolicy(Policy):
    @abstractmethod
    def random_action(
        self, obs: np.ndarray, h_state: np.ndarray, key: jrandom.PRNGKey
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def act_lprob(
        self, obs: np.ndarray, h_state: np.ndarray, key: jrandom.PRNGKey
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def lprob(
        self, obs: np.ndarray, h_state: np.ndarray, act: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class ActionValue(Model):
    @abstractmethod
    def q_values(
        self, x: np.ndarray, h_state: np.ndarray, act: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class Value(Model):
    @abstractmethod
    def values(
        self, x: np.ndarray, h_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class Encoder(Model):
    @abstractmethod
    def encode(
        self, x: np.ndarray, h_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass
