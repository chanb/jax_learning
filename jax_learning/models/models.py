from abc import abstractmethod
from typing import Tuple, Optional

import equinox as eqx
import jax.random as jrandom
import numpy as np


class Policy(eqx.Module):
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


class ActionValue(eqx.Module):
    @abstractmethod
    def q_values(self,
                 obs: np.ndarray,
                 h_state: np.ndarray,
                 act: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
        pass
