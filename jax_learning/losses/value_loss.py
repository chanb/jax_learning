import jax.numpy as jnp
import numpy as np


def one_step_bellman_optimality_error(q_curr_pred: np.ndarray,
                                      act: np.ndarray,
                                      q_next_pred: np.ndarray,
                                      rew: np.ndarray,
                                      done: np.ndarray,
                                      gamma: np.ndarray) -> np.ndarray:
    q_target = rew + (1 - done) * (gamma * jnp.max(q_next_pred))
    return q_curr_pred[act] - q_target
