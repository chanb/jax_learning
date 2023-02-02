from typing import Tuple, Sequence, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


class MLP(eqx.Module):
    _in_dim: int = eqx.static_field()
    _out_dim: int = eqx.static_field()
    weights: Sequence[eqx.nn.Linear]
    biases: Sequence[jnp.ndarray]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_hidden: int,
        key: jrandom.PRNGKey,
    ):
        self._in_dim = in_dim
        self._out_dim = out_dim
        if num_hidden == 0:
            self.weights = [eqx.nn.Linear(in_dim, out_dim, use_bias=False, key=key)]
            self.biases = [jnp.zeros(out_dim)]
            return

        self.weights = [eqx.nn.Linear(in_dim, hidden_dim, use_bias=False, key=key)]
        self.biases = [jnp.zeros(hidden_dim)]
        for _ in range(num_hidden - 1):
            key, _ = jrandom.split(key, num=2)
            self.weights.append(
                eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=key)
            )
            self.biases.append(jnp.zeros(hidden_dim))

        key, _ = jrandom.split(key, num=2)
        self.weights.append(eqx.nn.Linear(hidden_dim, out_dim, use_bias=False, key=key))
        self.biases.append(jnp.zeros(out_dim))

    @property
    def in_dim(self):
        return self._in_dim

    @property
    def out_dim(self):
        return self._out_dim

    @property
    def num_hidden(self):
        return len(self.weights) - 1

    @jax.jit
    def __call__(self, input: np.ndarray) -> np.ndarray:
        x = input
        for layer_i in range(self.num_hidden):
            x = jax.nn.relu(self.weights[layer_i](x) + self.biases[layer_i])
        x = self.weights[-1](x) + self.biases[-1]
        return x


class Conv2D(eqx.Module):
    # TODO: Add BatchNorm and Dropout
    parameters: Sequence[eqx.nn.Conv2d]
    activations: Sequence[Callable] = eqx.static_field()
    _dim_per_layer: Sequence[Tuple[int, int]] = eqx.static_field()

    def __init__(
        self,
        layers: Tuple[
            int,
            int,
            Sequence[int],
            Sequence[int],
            Sequence[int],
            Sequence[int],
            Callable,
            bool,
        ],
        in_dim: Tuple[int, int],
        key: jrandom.PRNGKey,
    ):
        assert (
            len(layers) > 0
        ), f"Number of CNN layers should be at least 1, got: {len(layers)}"
        self.parameters = []
        self.activations = []
        self._dim_per_layer = [in_dim]
        for (
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            use_bias,
        ) in layers:
            (height, width) = self._dim_per_layer[-1]
            key, _ = jrandom.split(key, num=2)
            self.parameters.append(
                eqx.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    use_bias=use_bias,
                    key=key,
                )
            )
            self.activations.append(activation)

            height = int(
                np.floor(
                    1
                    + float(
                        height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
                    )
                    / float(stride[0])
                )
            )
            width = int(
                np.floor(
                    1
                    + float(
                        width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
                    )
                    / float(stride[1])
                )
            )
            self._dim_per_layer.append((height, width))

    @property
    def dim_per_layer(self):
        return self._dim_per_layer

    @property
    def num_layers(self):
        return len(self.dim_per_layer) - 1

    @property
    def out_dim(self):
        return (self.parameters[-1].out_channels, *self._dim_per_layer[-1])

    @jax.jit
    def __call__(self, input: np.ndarray) -> np.ndarray:
        x = input
        for layer_i in range(self.num_layers):
            x = self.parameters[layer_i](x)
            x = self.activations[layer_i](x)
        return x


class SelfAttention(eqx.Module):
    _num_heads: int = eqx.static_field()
    _embd_dim: int = eqx.static_field()
    parameters: eqx.nn.MultiheadAttention

    def __init__(self, num_heads: int, embd_dim: int, key=jrandom.PRNGKey):
        self._embd_dim = embd_dim
        self._num_heads = num_heads
        self.parameters = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=embd_dim, key=key
        )

    @jax.jit
    def __call__(
        self, input: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        x = input
        x = self.parameters(query=x, key_=x, value=x, mask=mask, inference=True)
        return x


class CausalSelfAttention(SelfAttention):
    def __init__(self, num_heads: int, embd_dim: int, key=jrandom.PRNGKey):
        super().__init__(num_heads, embd_dim, key)

    @jax.jit
    def __call__(self, input: np.ndarray) -> np.ndarray:
        causal_mask = jnp.tril(
            np.ones((self._num_heads, input.shape[0], input.shape[0]))
        )
        return super().__call__(input, mask=causal_mask)


class GPT2Block(eqx.Module):
    attention: SelfAttention
    embedding: eqx.nn.Linear
    feedforward: eqx.nn.linear
    projection: eqx.nn.linear
    layer_norm_1: eqx.nn.LayerNorm
    layer_norm_2: eqx.nn.LayerNorm
    _in_dim: int = eqx.static_field()
    _vmap_feedforward: Callable = eqx.static_field()

    def __init__(self, in_dim: int, num_heads: int, embd_dim: int, key=jrandom.PRNGKey):
        self._in_dim = in_dim
        self.attention = CausalSelfAttention(
            num_heads=num_heads, embd_dim=embd_dim, key=key
        )
        key, _ = jrandom.split(key)
        self.feedforward = eqx.nn.Linear(embd_dim, embd_dim * 4, key=key)
        key, _ = jrandom.split(key)
        self.projection = eqx.nn.Linear(embd_dim * 4, embd_dim, key=key)
        key, _ = jrandom.split(key)
        self.embedding = eqx.nn.Linear(in_dim, embd_dim, key=key)
        self.layer_norm_1 = eqx.nn.LayerNorm(embd_dim)
        self.layer_norm_2 = eqx.nn.LayerNorm(embd_dim)
        self._vmap_feedforward = jax.vmap(self._feedforward)

    @jax.jit
    def __call__(self, input: np.ndarray) -> np.ndarray:
        x = jax.vmap(self.embedding)(input)
        x = x + self.attention(self.layer_norm_1(x))
        x = self._vmap_feedforward(x)
        return x

    def _feedforward(self, input: np.ndarray) -> np.ndarray:
        return input + self.projection(
            jax.nn.gelu(self.feedforward(self.layer_norm_2(input)))
        )
