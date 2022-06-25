from typing import Sequence, Tuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from jax_learning.models.models import Policy, ActionValue, MLP


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
        return act, h_state
    
    def random_action(self,
                      obs: np.ndarray,
                      h_state: np.ndarray,
                      key: jrandom.PRNGKey) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_val, h_state = self.q_values(obs, h_state)
        acts = jrandom.categorical(key=key, logits=q_val, axis=-1)
        return acts, h_state


class MLPSoftmaxQ(SoftmaxQ):
    q_function: eqx.Module

    def __init__(self,
                 obs_dim: Sequence[int],
                 act_dim: Sequence[int],
                 hidden_dim: int,
                 num_hidden: int,
                 key: jrandom.PRNGKey):
        super().__init__(obs_dim, act_dim)
        self.q_function = MLP(self.obs_dim, self.act_dim, hidden_dim, num_hidden, key)

    def q_values(self,
                 obs: np.ndarray,
                 h_state: np.ndarray,
                 act: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
        qs = self.q_function(obs)
        return qs, h_state
