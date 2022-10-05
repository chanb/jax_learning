import numpy as np


def squared_loss(pred: np.ndarray, targ: np.ndarray) -> np.ndarray:
    return (pred - targ) ** 2
