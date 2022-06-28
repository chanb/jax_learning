import numpy as np


def sac_temperature_loss(
    temp: float, lprob: np.ndarray, target_entropy: float
) -> np.ndarray:
    return temp * -(lprob + target_entropy)
