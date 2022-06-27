from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, Tuple

import numpy as np


class NoSampleError(Exception):
    pass


class LengthMismatchError(Exception):
    pass


class CheckpointIndexError(Exception):
    pass


class ReplayBuffer(ABC):
    @abstractproperty
    def buffer_size(self):
        raise NotImplementedError

    @abstractproperty
    def is_full(self):
        raise NotImplementedError

    @abstractproperty
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def push(
        self,
        obs: np.ndarray,
        h_state: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        info: dict,
        **kwargs
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def clear(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def sample(
        self, batch_size: int, idxes: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict,
        np.ndarray,
        np.ndarray,
    ]:
        raise NotImplementedError

    @abstractmethod
    def sample_init_obs(
        self, batch_size: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def save(self, save_path: str, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load(self, load_path: str, **kwargs):
        raise NotImplementedError
