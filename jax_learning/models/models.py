from abc import abstractmethod
from typing import Tuple

import equinox as eqx
import jax.random as jrandom
import numpy as np


class RLModel(eqx.Module):
    @abstractmethod
    def deterministic_action(self,
                             obs: np.ndarray,
                             h_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
    
    @abstractmethod
    def random_action(self,
                      obs: np.ndarray,
                      h_state: np.ndarray,
                      key: jrandom.PRNGKey) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
