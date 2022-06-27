import jax.numpy as jnp
import numpy as np


def monte_carlo_returns(rews: np.ndarray,
                        dones: np.ndarray,
                        gamma: float) -> np.ndarray:
    rets = np.zeros(rews.shape[0] + 1)
    for step in reversed(range(len(rews))):
        rets[step] = rets[step + 1] * gamma * (1 - dones[step]) + rews[step]
    return rets[:-1]

def q_learning_td_error(curr_q_pred: np.ndarray,
                        act: np.ndarray,
                        next_q_pred: np.ndarray,
                        rew: np.ndarray,
                        done: np.ndarray,
                        gamma: float) -> np.ndarray:
    q_target = rew + (1 - done) * (gamma * jnp.max(next_q_pred))
    return curr_q_pred[act] - q_target
