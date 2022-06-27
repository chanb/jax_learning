import numpy as np


def reinforce_loss(lprob: np.ndarray, ret: np.ndarray) -> np.ndarray:
    return -lprob * ret


def sac_policy_loss(
    curr_q_pred_min: np.ndarray, lprob: np.ndarray, temp: float
) -> np.ndarray:
    return -(curr_q_pred_min - temp * lprob)
