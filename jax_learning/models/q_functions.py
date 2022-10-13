from typing import Sequence, Tuple, Optional, Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from jax_learning.distributions import Categorical
from jax_learning.models import StochasticPolicy, ActionValue, MLP, Encoder
from jax_learning.models.encoders import NatureCNN


class MultiQ(ActionValue):
    qs: ActionValue

    def __init__(
        self,
        q_constructor: Callable[[jrandom.PRNGKey], ActionValue],
        num_qs: int,
        key: jrandom.PRNGKey,
    ):
        @eqx.filter_vmap(out=lambda x: 0 if eqx.is_array(x) else None)
        def make_qs(key):
            return q_constructor(key=key)

        self.qs = make_qs(jrandom.split(key, num_qs))

    @staticmethod
    @eqx.filter_vmap(kwargs=dict(obs=None, h_state=None, act=None))
    def _q_values(
        q: ActionValue, obs: np.ndarray, h_state: np.ndarray, act: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return q.q_values(obs, h_state, act)

    def q_values(
        self, obs: np.ndarray, h_state: np.ndarray, act: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        q_vals, h_states = self._q_values(self.qs, obs, h_state, act)
        return q_vals, h_states


class SoftmaxQ(StochasticPolicy, ActionValue):
    q_function: ActionValue

    def __init__(self, q_function: ActionValue):
        self.q_function = q_function

    def deterministic_action(
        self, obs: np.ndarray, h_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_val, h_state = self.q_values(obs, h_state)
        act = jnp.argmax(q_val, axis=-1)
        return act, h_state

    def random_action(
        self, obs: np.ndarray, h_state: np.ndarray, key: jrandom.PRNGKey
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_val, h_state = self.q_values(obs, h_state)
        act = Categorical.sample(q_val, key)
        return act, h_state

    def act_lprob(
        self, obs: np.ndarray, h_state: np.ndarray, key: jrandom.PRNGKey
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_val, h_state = self.q_values(obs, h_state)
        act = Categorical.sample(q_val, key)
        lprob = Categorical.lprob(q_val, act)
        return act, lprob, h_state

    def q_values(
        self, obs: np.ndarray, h_state: np.ndarray, act: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.q_function.q_values(obs, h_state, act)

    def dist_params(self, obs: np.ndarray, h_state: np.ndarray) -> Sequence[np.ndarray]:
        return self.q_function.q_values(obs, h_state)[0]

    def lprob(
        self, obs: np.ndarray, h_state: np.ndarray, act: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        q_val, h_state = self.q_values(obs, h_state)
        lprob = Categorical.lprob(q_val, act)
        return lprob, h_state


class MLPQ(ActionValue):
    in_dim: int = eqx.static_field()
    out_dim: int = eqx.static_field()
    q_function: eqx.Module

    def __init__(
        self,
        in_dim: Sequence[int],
        out_dim: Sequence[int],
        hidden_dim: int,
        num_hidden: int,
        key: jrandom.PRNGKey,
    ):
        self.in_dim = int(np.product(in_dim))
        self.out_dim = int(np.product(out_dim))
        self.q_function = MLP(self.in_dim, self.out_dim, hidden_dim, num_hidden, key)

    def q_values(
        self, obs: np.ndarray, h_state: np.ndarray, act: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = obs
        if act is not None:
            x = jnp.concatenate((obs, act), axis=-1)
        q_val = self.q_function(x)
        return q_val, h_state


class EncoderQ(ActionValue):
    encoder: Encoder
    q_function: ActionValue

    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        out_dim: Sequence[int],
        hidden_dim: int,
        num_hidden: int,
        key: jrandom.PRNGKey,
    ):
        self.encoder = NatureCNN(
            in_channels=in_channels, height=height, width=width, key=key
        )
        self.q_function = MLPQ(
            in_dim=self.encoder.encoder.dim_per_layer[-1],
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
        )

    def q_values(
        self, obs: np.ndarray, h_state: np.ndarray, act: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = obs
        x, h_state = self.encoder.encode(x, h_state)
        q_val = self.q_function(x.reshape(-1), h_state, act)
        return q_val, h_state
