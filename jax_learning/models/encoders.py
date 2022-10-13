import equinox as eqx
import jax
import jax.random as jrandom
import numpy as np

from typing import Tuple

from jax_learning.models.models import Encoder, Conv2D


class NatureCNN(Encoder):
    in_dim: Tuple[int, int, int] = eqx.static_field()
    encoder: Conv2D

    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        key: jrandom.PRNGKey,
    ):
        self.in_dim = (in_channels, height, width)
        layers = [
            [
                in_channels,
                32,
                (8, 8),
                (4, 4),
                (0, 0),
                (1, 1),
                jax.nn.relu,
                False,
            ],
            [32, 64, (4, 4), (2, 2), (0, 0), (1, 1), jax.nn.relu, False],
            [64, 64, (3, 3), (1, 1), (0, 0), (1, 1), jax.nn.relu, False],
        ]
        self.encoder = Conv2D(layers=layers, in_dim=(height, width), key=key)

    def encode(
        self, x: np.ndarray, h_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.encoder(x), h_state
