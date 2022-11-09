from gym import Env
from gym.spaces import Box
from gym.wrappers.transform_observation import TransformObservation

import numpy as np


class HWC2CHW(TransformObservation):
    def __init__(self, env: Env, scale: float = None):
        def transpose_and_normalize(input: np.ndarray):
            res = np.transpose(input, axes=(2, 0, 1))

            if scale is not None:
                return res.astype(np.float32) / scale
            return res

        super().__init__(env, transpose_and_normalize)

        dtype = self.observation_space.dtype
        low = self.observation_space.low
        high = self.observation_space.high
        if scale is not None:
            dtype = np.float32
            low.astype(dtype) / scale
            high.astype(dtype) / scale

        self.observation_space = Box(
            low=np.transpose(low, axes=(2, 0, 1)),
            high=np.transpose(high, axes=(2, 0, 1)),
            dtype=dtype,
        )
