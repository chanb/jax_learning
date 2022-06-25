from abc import abstractmethod
from typing import Tuple, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


class Policy(eqx.Module):
    @abstractmethod
    def deterministic_action(self,
                             obs: np.ndarray,
                             h_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass


class StochasticPolicy(Policy):
    @abstractmethod
    def random_action(self,
                      obs: np.ndarray,
                      h_state: np.ndarray,
                      key: jrandom.PRNGKey) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def act_lprob(self,
                  obs: np.ndarray,
                  h_state: np.ndarray,
                  key: jrandom.PRNGKey) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class ActionValue(eqx.Module):
    @abstractmethod
    def q_values(self,
                 obs: np.ndarray,
                 h_state: np.ndarray,
                 act: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
        pass


class MLP(eqx.Module):
    weights: Sequence[eqx.nn.Linear]
    biases: Sequence[jnp.ndarray]

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 num_hidden: int,
                 key: jrandom.PRNGKey):
        self.weights = [eqx.nn.Linear(in_dim, hidden_dim, use_bias=False, key=key)]
        self.biases = [jnp.zeros(hidden_dim)]
        for _ in range(num_hidden - 1):
            key, _ = jrandom.split(key, num=2)
            self.weights.append(eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=key))
            self.biases.append(jnp.zeros(hidden_dim))
            
        key, _ = jrandom.split(key, num=2)
        self.weights.append(eqx.nn.Linear(hidden_dim, out_dim, use_bias=False, key=key))
        self.biases.append(jnp.zeros(out_dim))

    @property
    def in_dim(self):
        return self.weights[0].in_features

    @property
    def out_dim(self):
        return self.weights[-1].out_features

    @property
    def num_hidden(self):
        return len(self.weights) - 1

    @jax.jit
    def __call__(self,
                 input: np.ndarray) -> np.ndarray:
        x = input
        for layer_i in range(self.num_hidden):
            x = jax.nn.relu(self.weights[layer_i](x) + self.biases[layer_i])
        x = self.weights[-1](x) + self.biases[-1]
        return x
