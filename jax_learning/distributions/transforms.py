from abc import abstractmethod

import equinox as eqx
import jax
import math
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


class TanhTransform(Transform):
    def transform(self,
                  x: np.ndarray) -> np.ndarray:
        return jax.nn.tanh(x)

    def log_abs_det_jacobian(self,
                             x: np.ndarray,
                             x_t: np.ndarray) -> np.ndarray:
        return 2. * (math.log(2.) - x - jax.nn.softplus(-2. * x))
