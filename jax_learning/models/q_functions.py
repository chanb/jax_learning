from typing import Sequence, Tuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from jax_learning.models.models import Policy, ActionValue


class SoftmaxQ(Policy, ActionValue):
    obs_dim: int
    act_dim: int

    def __init__(self,
                 obs_dim: Sequence[int],
                 act_dim: Sequence[int]):
        self.obs_dim = int(np.product(obs_dim))
        self.act_dim = int(np.product(act_dim))

    def deterministic_action(self,
                             obs: np.ndarray,
                             h_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_val, h_state = self.q_values(obs, h_state)
        act = jnp.argmax(q_val, axis=-1)
        return act, q_val[act].reshape((1,)), h_state
    
    def random_action(self,
                      obs: np.ndarray,
                      h_state: np.ndarray,
                      key: jrandom.PRNGKey) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_val, h_state = self.q_values(obs, h_state)
        acts = jrandom.categorical(key=key, logits=q_val, axis=-1)
        return acts, q_val[acts].reshape((1,)), h_state
    

class MLPSoftmaxQ(SoftmaxQ):
    weights: Sequence[eqx.nn.Linear]
    biases: Sequence[jnp.ndarray]

    @property
    def num_hidden(self):
        return len(self.weights) - 1

    def __init__(self,
                 obs_dim: Sequence[int],
                 act_dim: Sequence[int],
                 hidden_dim: int,
                 num_hidden: int,
                 key: jrandom.PRNGKey):
        super().__init__(obs_dim, act_dim)
        self.weights = [eqx.nn.Linear(self.obs_dim, hidden_dim, use_bias=False, key=key)]
        self.biases = [jnp.zeros(hidden_dim)]
        for _ in range(num_hidden - 1):
            key, _ = jrandom.split(key, num=2)
            self.weights.append(eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=key))
            self.biases.append(jnp.zeros(hidden_dim))
            
        key, _ = jrandom.split(key, num=2)
        self.weights.append(eqx.nn.Linear(hidden_dim, self.act_dim, use_bias=False, key=key))
        self.biases.append(jnp.zeros(self.act_dim))

    def q_values(self,
                 obs: np.ndarray,
                 h_state: np.ndarray,
                 act: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
        x = obs
        for layer_i in range(self.num_hidden):
            x = jax.nn.relu(self.weights[layer_i](x) + self.biases[layer_i])
        x = self.weights[-1](x) + self.biases[-1]
        return x, h_state
