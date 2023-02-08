import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom


class PositionalEmbedding1D(eqx.nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_size: int,
        key: jrandom.PRNGKey,
    ):
        pos = jnp.arange(num_embeddings)[:, None].astype(float)
        sin_denom = jnp.exp(-(jnp.log(jnp.array([10000.0])) / embedding_size) * 2 * jnp.arange(jnp.floor(embedding_size / 2)))
        cos_denom = jnp.exp(-(jnp.log(jnp.array([10000.0])) / embedding_size) * 2 * jnp.arange(jnp.ceil(embedding_size / 2)))
        weight = jnp.concatenate((jnp.sin(pos * sin_denom), jnp.cos(pos * cos_denom)), axis=1)
        print(weight.shape)
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_size=embedding_size,
            weight=weight,
            key=key
        )
