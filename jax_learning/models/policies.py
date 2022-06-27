from typing import Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from jax_learning.distributions import Distribution, Normal
from jax_learning.distributions.transforms import TanhTransform
from jax_learning.models import StochasticPolicy, MLP


class MLPGaussianPolicy(StochasticPolicy):
    obs_dim: int
    act_dim: int
    min_std: float
    policy: eqx.Module

    def __init__(self,
                 obs_dim: Sequence[int],
                 act_dim: Sequence[int],
                 hidden_dim: int,
                 num_hidden: int,
                 key: jrandom.PRNGKey,
                 min_std: float=1e-7):
        self.obs_dim = int(np.product(obs_dim))
        self.act_dim = int(np.product(act_dim))
        self.min_std = min_std
        self.policy = MLP(self.obs_dim, self.act_dim * 2, hidden_dim, num_hidden, key)

    def deterministic_action(self,
                             obs: np.ndarray,
                             h_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        act_mean, _ = jnp.split(self.policy(obs), 2, axis=-1)
        return act_mean, h_state
    
    def random_action(self,
                      obs: np.ndarray,
                      h_state: np.ndarray,
                      key: jrandom.PRNGKey) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dist = self.dist(obs, h_state)
        act = dist.sample(key)
        return act, h_state
    
    def act_lprob(self,
                  obs: np.ndarray,
                  h_state: np.ndarray,
                  key: jrandom.PRNGKey) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dist = self.dist(obs, h_state)
        act = dist.sample(key)
        lprob = dist.lprob(act)
        return act, lprob, h_state
    
    def dist(self,
             obs: np.ndarray,
             h_state: np.ndarray) -> Distribution:
        act_mean, act_raw_std = jnp.split(self.policy(obs), 2, axis=-1)
        act_std = jax.nn.softplus(act_raw_std) + self.min_std
        return Normal(act_mean, act_std)


class MLPSquashedGaussianPolicy(MLPGaussianPolicy):
    def __init__(self,
                 obs_dim: Sequence[int],
                 act_dim: Sequence[int],
                 hidden_dim: int,
                 num_hidden: int,
                 key: jrandom.PRNGKey,
                 min_std: float=1e-7):
        super().__init__(obs_dim, act_dim, hidden_dim, num_hidden, key, min_std)

    def deterministic_action(self,
                             obs: np.ndarray,
                             h_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        act_mean, _ = jnp.split(self.policy(obs), 2, axis=-1)
        act_t = TanhTransform.transform(act_mean)
        return act_t, h_state
    
    def random_action(self,
                      obs: np.ndarray,
                      h_state: np.ndarray,
                      key: jrandom.PRNGKey) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dist = self.dist(obs, h_state)
        act = dist.sample(key)
        act_t = TanhTransform.transform(act)
        return act_t, h_state
    
    def act_lprob(self,
                  obs: np.ndarray,
                  h_state: np.ndarray,
                  key: jrandom.PRNGKey) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dist = self.dist(obs, h_state)
        act = dist.sample(key)
        act_t = TanhTransform.transform(act)
        lprob = dist.lprob(act) - TanhTransform.log_abs_det_jacobian(act, act_t)
        return act, lprob, h_state
    
    def dist(self,
             obs: np.ndarray,
             h_state: np.ndarray) -> Distribution:
        act_mean, act_raw_std = jnp.split(self.policy(obs), 2, axis=-1)
        act_std = jax.nn.softplus(act_raw_std) + self.min_std
        return Normal(act_mean, act_std)
