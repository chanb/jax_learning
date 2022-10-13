from typing import Sequence, Tuple, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from jax_learning.models import Value, MLP


class MLPValue(Value):
    in_dim: int = eqx.static_field()
    out_dim: int = eqx.static_field()
    value_function: eqx.Module

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
        self.value_function = MLP(
            self.in_dim, self.out_dim, hidden_dim, num_hidden, key
        )

    def values(
        self, obs: np.ndarray, h_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        val = self.value_function(obs)
        return val, h_state
