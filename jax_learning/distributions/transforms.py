from abc import abstractmethod, abstractstaticmethod

import equinox as eqx
import jax
import math
import numpy as np


class Transform(eqx.Module):
    @abstractstaticmethod
    def transform(x: np.ndarray) -> np.ndarray:
        pass

    @abstractstaticmethod
    def log_abs_det_jacobian(self,
                             x: np.ndarray,
                             x_t: np.ndarray) -> np.ndarray:
        pass


class TanhTransform(Transform):
    def transform(x: np.ndarray) -> np.ndarray:
        return jax.nn.tanh(x)

    def log_abs_det_jacobian(x: np.ndarray,
                             x_t: np.ndarray) -> np.ndarray:
        return 2. * (math.log(2.) - x - jax.nn.softplus(-2. * x))
