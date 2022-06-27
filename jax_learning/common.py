from typing import Sequence, Callable

import numpy as np


def polyak_average_generator(
    x: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def polyak_average(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return x * p + (1 - x) * q

    return polyak_average


class RunningMeanStd:
    """Modified from Baseline
    Assumes shape to be (number of inputs, input_shape)
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        shape: Sequence[int] = (),
        a_min: float = -5.0,
        a_max: float = 5.0,
    ):
        assert epsilon > 0.0
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float)
        self.var = np.ones(shape, dtype=np.float)
        self.epsilon = epsilon
        self.count = 0
        self.a_min = a_min
        self.a_max = a_max

    def update(self, x: np.ndarray):
        x = x.reshape(-1, *self.shape)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = len(x)

        if batch_count == 0:
            return
        elif batch_count == 1:
            batch_var.fill(0.0)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta**2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x_shape = x.shape
        x = x.reshape(-1, *self.shape)
        normalized_x = np.clip(
            (x - self.mean) / np.sqrt(self.var + self.epsilon),
            a_min=self.a_min,
            a_max=self.a_max,
        )
        normalized_x[normalized_x != normalized_x] = 0.0
        normalized_x = normalized_x.reshape(x_shape)
        return normalized_x

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        x_shape = x.shape
        x = x.reshape(-1, *self.shape)
        return (x * np.sqrt(self.var + self.epsilon) + self.mean).reshape(x_shape)
