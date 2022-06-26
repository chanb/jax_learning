from abc import abstractmethod

import equinox as eqx
import numpy as np


class Transform(eqx.Module):
    @abstractmethod
    def transform(self,
                  x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def log_abs_det_jacobian(self,
                             x: np.ndarray,
                             x_t: np.ndarray) -> np.ndarray:
        pass


class Tanh(eqx.Module):
    def transform(self):
        pass
