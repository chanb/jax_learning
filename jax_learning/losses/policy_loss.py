import numpy as np


def reinforce_loss(lprob: np.ndarray, ret: np.ndarray) -> np.ndarray:
    return -lprob * ret
