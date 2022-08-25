from typing import Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from jax_learning.distributions import Normal, Categorical
from jax_learning.distributions.transforms import TanhTransform
from jax_learning.models import StochasticPolicy, MLP


class MLPSoftmaxPolicy(StochasticPolicy):
    obs_dim: int
    act_dim: int
    policy: eqx.Module

    def __init__(
        self,
        obs_dim: Sequence[int],
        act_dim: Sequence[int],
        hidden_dim: int,
        num_hidden: int,
        key: jrandom.PRNGKey,
    ):
        self.obs_dim = int(np.product(obs_dim))
        self.act_dim = int(np.product(act_dim))
        self.policy = MLP(self.obs_dim, self.act_dim, hidden_dim, num_hidden, key)

    def deterministic_action(
        self, obs: np.ndarray, h_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits = self.dist_params(obs, h_state)
        return jnp.argmax(logits, axis=-1), h_state

    def random_action(
        self, obs: np.ndarray, h_state: np.ndarray, key: jrandom.PRNGKey
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits = self.dist_params(obs, h_state)
        act = Categorical.sample(logits, key)
        return act, h_state

    def act_lprob(
        self, obs: np.ndarray, h_state: np.ndarray, key: jrandom.PRNGKey
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits = self.dist_params(obs, h_state)
        act = Categorical.sample(logits, key)
        lprob = Categorical.lprob(logits, act)
        return act, lprob, h_state

    def dist_params(self, obs: np.ndarray, h_state: np.ndarray) -> Sequence[np.ndarray]:
        return self.policy(obs)

    def lprob(
        self, obs: np.ndarray, h_state: np.ndarray, act: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        logits = self.dist_params(obs, h_state)
        lprob = Categorical.lprob(logits, act)
        return lprob, h_state


class MLPGaussianPolicy(StochasticPolicy):
    obs_dim: int
    act_dim: int
    min_std: float
    policy: eqx.Module

    def __init__(
        self,
        obs_dim: Sequence[int],
        act_dim: Sequence[int],
        hidden_dim: int,
        num_hidden: int,
        key: jrandom.PRNGKey,
        min_std: float = 1e-7,
    ):
        self.obs_dim = int(np.product(obs_dim))
        self.act_dim = int(np.product(act_dim))
        self.min_std = min_std
        self.policy = MLP(self.obs_dim, self.act_dim * 2, hidden_dim, num_hidden, key)

    def deterministic_action(
        self, obs: np.ndarray, h_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        act_mean, _ = self.dist_params(obs, h_state)
        return act_mean, h_state

    def random_action(
        self, obs: np.ndarray, h_state: np.ndarray, key: jrandom.PRNGKey
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        act_mean, act_std = self.dist_params(obs, h_state)
        act = Normal.sample(act_mean, act_std, key)
        return act, h_state

    def act_lprob(
        self, obs: np.ndarray, h_state: np.ndarray, key: jrandom.PRNGKey
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        act_mean, act_std = self.dist_params(obs, h_state)
        act = Normal.sample(act_mean, act_std, key)
        lprob = Normal.lprob(act_mean, act_std, act)
        return act, lprob, h_state

    def dist_params(self, obs: np.ndarray, h_state: np.ndarray) -> Sequence[np.ndarray]:
        act_mean, act_raw_std = jnp.split(self.policy(obs), 2, axis=-1)
        act_std = jax.nn.softplus(act_raw_std) + self.min_std
        return act_mean, act_std

    def lprob(
        self, obs: np.ndarray, h_state: np.ndarray, act: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        act_mean, act_std = self.dist_params(obs, h_state)
        lprob = Normal.lprob(act_mean, act_std, act)
        return lprob, h_state


class MLPSquashedGaussianPolicy(MLPGaussianPolicy):
    def __init__(
        self,
        obs_dim: Sequence[int],
        act_dim: Sequence[int],
        hidden_dim: int,
        num_hidden: int,
        key: jrandom.PRNGKey,
        min_std: float = 1e-7,
    ):
        super().__init__(obs_dim, act_dim, hidden_dim, num_hidden, key, min_std)

    def deterministic_action(
        self, obs: np.ndarray, h_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        act_mean, _ = self.dist_params(obs, h_state)
        act_t = TanhTransform.transform(act_mean)
        return act_t, h_state

    def random_action(
        self, obs: np.ndarray, h_state: np.ndarray, key: jrandom.PRNGKey
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        act_mean, act_std = self.dist_params(obs, h_state)
        act = Normal.sample(act_mean, act_std, key)
        act_t = TanhTransform.transform(act)
        return act_t, h_state

    def act_lprob(
        self, obs: np.ndarray, h_state: np.ndarray, key: jrandom.PRNGKey
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        act_mean, act_std = self.dist_params(obs, h_state)
        act = Normal.sample(act_mean, act_std, key)
        act_t = TanhTransform.transform(act)
        lprob = Normal.lprob(act_mean, act_std, act)
        lprob = lprob - TanhTransform.log_abs_det_jacobian(act, act_t)
        return act_t, lprob, h_state

    def lprob(
        self, obs: np.ndarray, h_state: np.ndarray, act: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        act_pret = jnp.arctanh(act)
        print(act, act_pret)
        act_mean, act_std = self.dist_params(obs, h_state)
        lprob = Normal.lprob(act_mean, act_std, act)
        lprob = lprob - TanhTransform.log_abs_det_jacobian(act_pret, act)
        return lprob, h_state
