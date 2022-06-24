from typing import Iterable

import jax


def to_jnp(*args: Iterable):
    return [jax.device_put(arg) for arg in args]

def batch_flatten(*args: Iterable):
    return [arg.reshape((len(arg), -1)) for arg in args]
