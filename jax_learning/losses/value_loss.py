import jax.numpy as jnp
import numpy as np


def monte_carlo_returns(
    rews: np.ndarray, dones: np.ndarray, gamma: float
) -> np.ndarray:
    rets = np.zeros(rews.shape[0] + 1)
    for step in reversed(range(len(rews))):
        rets[step] = rets[step + 1] * gamma * (1 - dones[step]) + rews[step]
    return rets[:-1]


def n_step_bootstrapped_returns(
    v_preds: np.ndarray,
    rews: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lambd: float = 1.0,
) -> np.ndarray:
    rets = np.zeros(v_preds.shape)
    for step in reversed(range(len(rews) - 1)):
        rets.at[step].set(
            ((1 - lambd) * v_preds[step + 1] + lambd * rets[step + 1])
            * gamma
            * (1 - dones[step])
            + rews[step]
        )
    return rets


def q_learning_td_error(
    curr_q_pred: np.ndarray,
    act: np.ndarray,
    next_q_pred: np.ndarray,
    rew: np.ndarray,
    done: np.ndarray,
    gamma: float,
) -> np.ndarray:
    q_target = rew + (1 - done) * (gamma * jnp.max(next_q_pred))
    return curr_q_pred[act] - q_target


def clipped_min_q_td_error(
    curr_q_pred: np.ndarray,
    next_q_pred_min: np.ndarray,
    next_lprob: np.ndarray,
    rew: np.ndarray,
    done: np.ndarray,
    temp: float,
    gamma: float,
) -> np.ndarray:
    v_next = next_q_pred_min - temp * next_lprob
    curr_q_target = rew + gamma * (1 - done) * v_next
    return curr_q_pred - curr_q_target


def path_consistency_error(
    lprobs: np.ndarray,
    v_preds: np.ndarray,
    rews: np.ndarray,
    length: np.ndarray,
    temp: np.ndarray,
    gamma: float,
) -> np.ndarray:
    ent_reg_rews = rews - temp * lprobs
    target = 0
    disc_rews = ent_reg_rews * (gamma ** np.arange(len(ent_reg_rews)))
    target = jnp.where(jnp.arange(disc_rews.shape[0]) < length, disc_rews, 0).sum()
    target += (length == len(ent_reg_rews)) * (gamma**length) * v_preds[length]
    return v_preds[0] - target[0]
