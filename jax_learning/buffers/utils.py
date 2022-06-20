import jax
import numpy as np

def to_jnp(*args):
    return [jax.device_put(arg) for arg in args]

def batch_flatten(*args):
    return [arg.reshape((len(arg), -1)) for arg in args]
