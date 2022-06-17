from typing import Optional, Tuple

import numpy as np


class NoSampleError(Exception):
    pass


class LengthMismatchError(Exception):
    pass


class CheckpointIndexError(Exception):
    pass


class ReplayBuffer:
    @property
    def memory_size(self):
        raise NotImplementedError

    @property
    def is_full(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def push(self,
             obs: np.ndarray,
             h_state: np.ndarray,
             act: np.ndarray,
             rew: float,
             done: bool,
             info: dict,
             **kwargs) -> bool:
        raise NotImplementedError

    def clear(self,
              **kwargs):
        raise NotImplementedError

    def sample(self,
               batch_size: int,
               idxes: Optional[np.ndarray]=None,
               **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def sample_init_obs(self,
                        batch_size: int,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def save(self,
             save_path: str,
             **kwargs):
        raise NotImplementedError

    def load(self,
             load_path: str,
             **kwargs):
        raise NotImplementedError
