from abc import abstractmethod
from typing import Any, Dict, Sequence, Optional, Tuple

import _pickle as pickle
import gzip
import numpy as np
import os

from jax_learning.buffers import (
    ReplayBuffer,
    CheckpointIndexError,
    NoSampleError,
)
import jax_learning.constants as c


class AbstractNumPyBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        obs_dim: Sequence[int],
        h_state_dim: Sequence[int],
        act_dim: Sequence[int],
        rew_dim: Sequence[int],
        infos: dict = dict(),
        burn_in_window: int = 0,
        padding_first: bool = False,
        checkpoint_interval: int = 0,
        checkpoint_path: Optional[str] = None,
        rng: np.random.RandomState = np.random.RandomState(),
        dtype: np.dtype = np.float32,
        load_buffer: str = None,
    ):
        if load_buffer is not None:
            self.load(load_buffer)
        else:
            self.rng = rng
            self._buffer_size = buffer_size
            self._dtype = dtype
            self.observations = np.zeros(shape=(buffer_size, *obs_dim), dtype=dtype)
            self.hidden_states = np.zeros(
                shape=(buffer_size, *h_state_dim), dtype=np.float32
            )
            self.actions = np.zeros(shape=(buffer_size, *act_dim), dtype=np.float32)
            self.rewards = np.zeros(shape=(buffer_size, *rew_dim), dtype=np.float32)
            self.dones = np.zeros(shape=(buffer_size, 1), dtype=np.float32)
            self.terminateds = np.zeros(shape=(buffer_size, 1), dtype=np.float32)
            self.truncateds = np.zeros(shape=(buffer_size, 1), dtype=np.float32)
            self.infos = dict()
            for info_name, (info_shape, info_dtype) in infos.items():
                self.infos[info_name] = np.zeros(
                    shape=(buffer_size, *info_shape), dtype=info_dtype
                )

            # This keeps track of the past X observations and hidden states for RNN
            self.burn_in_window = burn_in_window
            if burn_in_window > 0:
                self.padding_first = padding_first
                self.historic_observations = np.zeros(
                    shape=(burn_in_window, *obs_dim), dtype=dtype
                )
                self.historic_hidden_states = np.zeros(
                    shape=(burn_in_window, *h_state_dim), dtype=dtype
                )
                self.historic_dones = np.ones(
                    shape=(burn_in_window, 1), dtype=np.float32
                )
            self._pointer = 0
            self._count = 0

        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_idxes = np.ones(shape=self._buffer_size, dtype=bool)
        if checkpoint_path is not None and self._buffer_size >= checkpoint_interval > 0:
            self._checkpoint_path = checkpoint_path
            os.makedirs(checkpoint_path, exist_ok=True)
            self.checkpoint = self._checkpoint
            self._checkpoint_count = 0
        else:
            self.checkpoint = lambda: None

    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def is_full(self):
        return self._count >= self.buffer_size

    @property
    def pointer(self):
        return self._pointer

    def set_size(self, size: int):
        if size == self._buffer_size:
            return
        elif size < self._buffer_size:
            self._buffer_size = size
            self._pointer = min(self._buffer_size, self._pointer) % self._buffer_size
        elif size > self._buffer_size:
            if self.is_full:
                self._pointer = self._buffer_size
            self._buffer_size = size
        self._count = self._pointer

    def __getitem__(self, index: int) -> tuple:
        return (
            self.observations[index],
            self.hidden_states[index],
            self.actions[index],
            self.rewards[index],
            self.dones[index],
            self.terminateds[index],
            self.truncateds[index],
            {
                info_name: info_value[index]
                for info_name, info_value in self.infos.items()
            },
        )

    def __len__(self) -> int:
        return min(self._count, self.buffer_size)

    def _checkpoint(self):
        transitions_to_save = np.where(self._checkpoint_idxes == 0)[0]

        if len(transitions_to_save) == 0:
            return

        idx_diff = np.where(
            transitions_to_save
            - np.concatenate(([transitions_to_save[0] - 1], transitions_to_save[:-1]))
            > 1
        )[0]

        if len(idx_diff) > 1:
            raise CheckpointIndexError
        elif len(idx_diff) == 1:
            transitions_to_save = np.concatenate(
                (transitions_to_save[idx_diff[0] :], transitions_to_save[: idx_diff[0]])
            )

        with gzip.open(
            os.path.join(f"{self._checkpoint_path}", f"{self._checkpoint_count}.pkl"),
            "wb",
        ) as f:
            pickle.dump(
                {
                    c.OBSERVATIONS: self.observations[transitions_to_save],
                    c.HIDDEN_STATES: self.hidden_states[transitions_to_save],
                    c.ACTIONS: self.actions[transitions_to_save],
                    c.REWARDS: self.rewards[transitions_to_save],
                    c.DONES: self.dones[transitions_to_save],
                    c.TERMINATEDS: self.terminateds[transitions_to_save],
                    c.TRUNCATEDS: self.truncateds[transitions_to_save],
                    c.INFOS: {
                        info_name: info_value[transitions_to_save]
                        for info_name, info_value in self.infos.items()
                    },
                },
                f,
            )

        self._checkpoint_idxes[transitions_to_save] = 1
        self._checkpoint_count += 1

    def push(
        self,
        obs: np.ndarray,
        h_state: np.ndarray,
        act: np.ndarray,
        rew: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        **kwargs,
    ) -> bool:
        # Stores the overwritten observation and hidden state into historic variables
        if self.burn_in_window > 0 and self._count >= self._buffer_size:
            self.historic_observations = np.concatenate(
                (self.historic_observations[1:], [self.observations[self._pointer]])
            )
            self.historic_hidden_states = np.concatenate(
                (self.historic_hidden_states[1:], [self.hidden_states[self._pointer]])
            )
            self.historic_dones = np.concatenate(
                (self.historic_dones[1:], [self.dones[self._pointer]])
            )

        self.observations[self._pointer] = obs
        self.hidden_states[self._pointer] = h_state
        self.actions[self._pointer] = act
        self.rewards[self._pointer] = rew
        self.dones[self._pointer] = np.logical_or(terminated, truncated)
        self.terminateds[self._pointer] = terminated
        self.truncateds[self._pointer] = truncated
        self._checkpoint_idxes[self._pointer] = 0
        for info_name, info_value in info.items():
            if info_name not in self.infos:
                continue
            self.infos[info_name][self._pointer] = info_value

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._count += 1

        if (
            self._checkpoint_interval > 0
            and (self._buffer_size - self._checkpoint_idxes.sum())
            >= self._checkpoint_interval
        ):
            self.checkpoint()
        return True

    def clear(self, **kwargs):
        self._pointer = 0
        self._count = 0
        self._checkpoint_idxes.fill(1)
        if self.burn_in_window > 0:
            self.historic_observations.fill(0.0)
            self.historic_hidden_states.fill(0.0)
            self.historic_dones.fill(1)

    def _get_burn_in_window(
        self, idxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        historic_observations = np.zeros(
            (len(idxes), self.burn_in_window, *self.observations.shape[1:])
        )
        historic_hidden_states = np.zeros(
            (len(idxes), self.burn_in_window, *self.hidden_states.shape[1:])
        )
        not_dones = np.ones((len(idxes), self.burn_in_window), dtype=bool)
        lengths = np.zeros(len(idxes), dtype=np.int64)

        shifted_idxes = (
            (idxes - self._pointer) % len(self)
            - np.arange(self.burn_in_window, 0, -1)[:, None]
        ).T
        cyclic_idxes = (idxes - np.arange(self.burn_in_window, 0, -1)[:, None]).T

        # Determine which index needs to look into historic buffer
        historic_idxes = np.logical_and(
            shifted_idxes >= -self.burn_in_window, shifted_idxes < 0
        ).astype(np.int64)
        non_historic_idxes = 1 - historic_idxes

        # Check whether we have reached another episode
        not_dones[np.where(self.dones[cyclic_idxes, 0] * non_historic_idxes)] = 0
        not_dones[
            np.where(
                self.historic_dones[shifted_idxes * historic_idxes, 0] * historic_idxes
            )
        ] = 0

        lengths = (
            np.argmin(np.flip(not_dones, axis=1), axis=1)
            + np.all(not_dones, axis=1) * self.burn_in_window
        )
        take_mask = np.concatenate(
            (
                np.zeros((1, self.burn_in_window), dtype=np.int64),
                np.eye(self.burn_in_window),
            ),
            axis=0,
        )[lengths]
        np.cumsum(np.flip(take_mask, axis=1), axis=1, out=take_mask)

        non_historic_take_mask = np.where(take_mask * non_historic_idxes)
        non_historic_cyclic_idxes = cyclic_idxes[non_historic_take_mask]
        historic_take_mask = np.where(take_mask * historic_idxes)
        historic_cyclic_idxes = cyclic_idxes[historic_take_mask]

        # Update for transitions not looking into historic buffer
        historic_observations[non_historic_take_mask] = self.observations[
            non_historic_cyclic_idxes
        ]
        historic_hidden_states[non_historic_take_mask] = self.hidden_states[
            non_historic_cyclic_idxes
        ]

        # Update for transitions looking into historic buffer
        if self._count > self._buffer_size:
            historic_observations[historic_take_mask] = self.observations[
                historic_cyclic_idxes
            ]
            historic_hidden_states[historic_take_mask] = self.hidden_states[
                historic_cyclic_idxes
            ]

        return historic_observations, historic_hidden_states, lengths

    def get_transitions(
        self, idxes: np.ndarray
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict,
        np.ndarray,
    ]:
        obss = self.observations[idxes]
        h_states = self.hidden_states[idxes]
        acts = self.actions[idxes]
        rews = self.rewards[idxes]
        dones = self.dones[idxes]
        terminateds = self.terminateds[idxes]
        truncateds = self.truncateds[idxes]
        infos = {
            info_name: info_value[idxes] for info_name, info_value in self.infos.items()
        }

        lengths = np.ones(len(obss), dtype=np.int64)
        if self.burn_in_window:
            historic_obss, historic_h_states, lengths = self._get_burn_in_window(idxes)
            obss = np.concatenate((historic_obss, obss[:, None, ...]), axis=1)
            h_states = np.concatenate(
                (historic_h_states, h_states[:, None, ...]), axis=1
            )
            lengths += 1

            # NOTE: Bad naming but padding first means non-zero entries in the beginning, then pad with zero afterwards
            if self.padding_first:
                obss = np.array(
                    [
                        np.roll(
                            history, shift=(length % (self.burn_in_window + 1)), axis=0
                        )
                        for history, length in zip(obss, lengths)
                    ]
                )
                h_states = np.array(
                    [
                        np.roll(
                            history, shift=(length % (self.burn_in_window + 1)), axis=0
                        )
                        for history, length in zip(h_states, lengths)
                    ]
                )
        else:
            obss = obss[:, None, ...]
            h_states = h_states[:, None, ...]

        return (
            obss,
            h_states,
            acts,
            rews,
            dones,
            terminateds,
            truncateds,
            infos,
            lengths,
        )

    @abstractmethod
    def sample(
        self, batch_size: int, idxes: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
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

    def sample_init_obs(
        self, batch_size: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.sum(self.dones) == 0:
            raise NoSampleError
        init_idxes = (np.where(self.dones == 1)[0] + 1) % self._buffer_size

        # Filter out indices greater than the current pointer.
        # It is useless once we exceed count > memory size
        init_idxes = init_idxes[init_idxes < len(self)]

        # Filter out the sample at is being pointed because we can't tell confidently that it is initial state
        init_idxes = init_idxes[init_idxes != self._pointer]
        random_idxes = init_idxes[self.rng.randint(len(init_idxes), size=batch_size)]
        return self.observations[random_idxes], self.hidden_states[random_idxes]

    def get_buffer_dict(self) -> Dict[str, Any]:
        buffer_dict = {
            c.OBSERVATIONS: self.observations,
            c.HIDDEN_STATES: self.hidden_states,
            c.ACTIONS: self.actions,
            c.REWARDS: self.rewards,
            c.DONES: self.dones,
            c.TERMINATEDS: self.terminateds,
            c.TRUNCATEDS: self.truncateds,
            c.INFOS: self.infos,
            c.BUFFER_SIZE: self._buffer_size,
            c.DTYPE: self._dtype,
            c.RNG: self.rng,
            c.BURN_IN_WINDOW: {
                c.BURN_IN_WINDOW: self.burn_in_window,
            },
        }
        if self.burn_in_window > 0:
            buffer_dict[c.BURN_IN_WINDOW][c.OBSERVATIONS] = self.historic_observations
            buffer_dict[c.BURN_IN_WINDOW][c.HIDDEN_STATES] = self.historic_hidden_states
            buffer_dict[c.BURN_IN_WINDOW][c.DONES] = self.historic_dones

        return buffer_dict

    def load_from_buffer_dict(self, buffer_dict: Dict[str, Any]):
        self._buffer_size = buffer_dict[c.BUFFER_SIZE]
        self.observations = buffer_dict[c.OBSERVATIONS]
        self.hidden_states = buffer_dict[c.HIDDEN_STATES]
        self.actions = buffer_dict[c.ACTIONS]
        self.rewards = buffer_dict[c.REWARDS]
        self.dones = buffer_dict[c.DONES]
        self.terminateds = buffer_dict[c.TERMINATEDS]
        self.truncateds = buffer_dict[c.TRUNCATEDS]
        self.infos = buffer_dict[c.INFOS]

        self._pointer = buffer_dict[c.POINTER]
        self._count = buffer_dict[c.COUNT]

        self._dtype = buffer_dict[c.DTYPE]
        self.rng = buffer_dict[c.RNG]

        if c.BURN_IN_WINDOW in buffer_dict:
            self.burn_in_window = buffer_dict[c.BURN_IN_WINDOW][c.BURN_IN_WINDOW]

            if self.burn_in_window > 0:
                self.historic_observations = buffer_dict[c.BURN_IN_WINDOW][
                    c.OBSERVATIONS
                ]
                self.historic_hidden_states = buffer_dict[c.BURN_IN_WINDOW][
                    c.HIDDEN_STATES
                ]
                self.historic_dones = buffer_dict[c.BURN_IN_WINDOW][c.DONES]

    def save(self, save_path: str, end_with_done: bool = True, **kwargs):
        pointer = self._pointer
        count = self._count

        if end_with_done:
            done_idxes = np.where(self.dones == 1)[0]
            if len(done_idxes) == 0:
                print("No completed episodes. Nothing to save.")
                return

            wraparound_idxes = done_idxes[done_idxes < self._pointer]
            if len(wraparound_idxes) > 0:
                pointer = (wraparound_idxes[-1] + 1) % self._buffer_size
                count -= self._pointer - pointer
            else:
                pointer = (done_idxes[-1] + 1) % self._buffer_size
                count -= self._pointer + self._buffer_size - pointer

        buffer_dict = self.get_buffer_dict()
        buffer_dict[c.POINTER] = pointer
        buffer_dict[c.COUNT] = count

        with gzip.open(save_path, "wb") as f:
            pickle.dump(
                buffer_dict,
                f,
            )

    def load(self, load_path: str, **kwargs):
        with gzip.open(load_path, "rb") as f:
            buffer_dict = pickle.load(f)
        self.load_from_buffer_dict(buffer_dict)


class TransitionNumPyBuffer(AbstractNumPyBuffer):
    @abstractmethod
    def get_next(self, next_idxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def sample(
        self, batch_size: int, idxes: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict,
        np.ndarray,
        np.ndarray,
    ]:
        if not len(self):
            raise NoSampleError

        if idxes is None:
            random_idxes = self.rng.randint(len(self), size=batch_size)
        else:
            random_idxes = idxes

        (
            obss,
            h_states,
            acts,
            rews,
            dones,
            terminateds,
            truncateds,
            infos,
            lengths,
        ) = self.get_transitions(random_idxes)

        return (
            obss,
            h_states,
            acts,
            rews,
            dones,
            terminateds,
            truncateds,
            infos,
            lengths,
            random_idxes,
        )

    def sample_with_next_obs(
        self,
        batch_size: int,
        idxes: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict,
        np.ndarray,
        np.ndarray,
    ]:
        (
            obss,
            h_states,
            acts,
            rews,
            dones,
            terminateds,
            truncateds,
            infos,
            lengths,
            random_idxes,
        ) = TransitionNumPyBuffer.sample(self, batch_size, idxes)

        next_idxes = random_idxes + 1
        next_obss, next_h_states = self.get_next(next_idxes)
        next_obss = next_obss[:, None, ...]
        next_h_states = next_h_states[:, None, ...]

        return (
            obss,
            h_states,
            acts,
            rews,
            dones,
            terminateds,
            truncateds,
            next_obss,
            next_h_states,
            infos,
            lengths,
            random_idxes,
        )


class MemoryEfficientNumPyBuffer(TransitionNumPyBuffer):
    def __init__(
        self,
        buffer_size: int,
        obs_dim: Sequence[int],
        h_state_dim: Sequence[int],
        act_dim: Sequence[int],
        rew_dim: Sequence[int],
        infos: dict = dict(),
        burn_in_window: int = 0,
        padding_first: bool = False,
        checkpoint_interval: int = 0,
        checkpoint_path: Optional[str] = None,
        rng: np.random.RandomState = np.random.RandomState(),
        dtype: np.dtype = np.float32,
        load_buffer: str = None,
    ):
        self.next_observation = None
        self.next_hidden_state = None
        super().__init__(
            buffer_size=buffer_size,
            obs_dim=obs_dim,
            h_state_dim=h_state_dim,
            act_dim=act_dim,
            rew_dim=rew_dim,
            infos=infos,
            burn_in_window=burn_in_window,
            padding_first=padding_first,
            checkpoint_interval=checkpoint_interval,
            checkpoint_path=checkpoint_path,
            rng=rng,
            dtype=dtype,
            load_buffer=load_buffer,
        )

    def push(
        self,
        obs: np.ndarray,
        h_state: np.ndarray,
        act: np.ndarray,
        rew: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        next_obs: np.ndarray,
        next_h_state: np.ndarray,
        **kwargs,
    ) -> bool:
        self.next_observation = next_obs
        self.next_hidden_state = next_h_state

        return super().push(
            obs=obs,
            h_state=h_state,
            act=act,
            rew=rew,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def get_next(self, next_idxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Replace all indices that are equal to memory size to zero
        idxes_to_modify = np.where(next_idxes == len(self))[0]
        next_idxes[idxes_to_modify] = 0
        next_obss = self.observations[next_idxes]
        next_h_states = self.hidden_states[next_idxes]

        # Replace the content for the sample at current timestep
        idxes_to_modify = np.where(next_idxes == self._pointer)[0]
        next_obss[idxes_to_modify] = self.next_observation
        next_h_states[idxes_to_modify] = self.next_hidden_state

        return next_obss, next_h_states

    def get_buffer_dict(self) -> Dict[str, Any]:
        buffer_dict = super().get_buffer_dict()
        buffer_dict[c.NEXT_HIDDEN_STATE] = self.next_hidden_state
        buffer_dict[c.NEXT_OBSERVATION] = self.next_observation
        return buffer_dict

    def load_from_buffer_dict(self, buffer_dict: Dict[str, Any]):
        super().load_from_buffer_dict(buffer_dict)
        self.next_observation = buffer_dict[c.NEXT_OBSERVATION]
        self.next_hidden_state = buffer_dict[c.NEXT_HIDDEN_STATE]


class NextStateNumPyBuffer(TransitionNumPyBuffer):
    def __init__(
        self,
        buffer_size: int,
        obs_dim: Sequence[int],
        h_state_dim: Sequence[int],
        act_dim: Sequence[int],
        rew_dim: Sequence[int],
        infos: dict = dict(),
        burn_in_window: int = 0,
        padding_first: bool = False,
        checkpoint_interval: int = 0,
        checkpoint_path: Optional[str] = None,
        rng: np.random.RandomState = np.random.RandomState(),
        dtype: np.dtype = np.float32,
        load_buffer: str = None,
    ):
        self.next_observations = np.zeros(shape=(buffer_size, *obs_dim), dtype=dtype)
        self.next_hidden_states = np.zeros(
            shape=(buffer_size, *h_state_dim), dtype=np.float32
        )
        super().__init__(
            buffer_size=buffer_size,
            obs_dim=obs_dim,
            h_state_dim=h_state_dim,
            act_dim=act_dim,
            rew_dim=rew_dim,
            infos=infos,
            burn_in_window=burn_in_window,
            padding_first=padding_first,
            checkpoint_interval=checkpoint_interval,
            checkpoint_path=checkpoint_path,
            rng=rng,
            dtype=dtype,
            load_buffer=load_buffer,
        )

    def __getitem__(self, index: int):
        return (
            self.observations[index],
            self.hidden_states[index],
            self.actions[index],
            self.rewards[index],
            self.dones[index],
            self.terminateds[index],
            self.truncateds[index],
            self.next_observations[index],
            self.next_hidden_states[index],
            {
                info_name: info_value[index]
                for info_name, info_value in self.infos.items()
            },
        )

    def _checkpoint(self):
        transitions_to_save = np.where(self._checkpoint_idxes == 0)[0]

        if len(transitions_to_save) == 0:
            return

        idx_diff = np.where(
            transitions_to_save
            - np.concatenate(([transitions_to_save[0] - 1], transitions_to_save[:-1]))
            > 1
        )[0]

        if len(idx_diff) > 1:
            raise CheckpointIndexError
        elif len(idx_diff) == 1:
            transitions_to_save = np.concatenate(
                (transitions_to_save[idx_diff[0] :], transitions_to_save[: idx_diff[0]])
            )

        with gzip.open(
            os.path.join(f"{self._checkpoint_path}", f"{self._checkpoint_count}.pkl"),
            "wb",
        ) as f:
            pickle.dump(
                {
                    c.OBSERVATIONS: self.observations[transitions_to_save],
                    c.HIDDEN_STATES: self.hidden_states[transitions_to_save],
                    c.ACTIONS: self.actions[transitions_to_save],
                    c.REWARDS: self.rewards[transitions_to_save],
                    c.DONES: self.dones[transitions_to_save],
                    c.TERMINATEDS: self.terminateds[transitions_to_save],
                    c.TRUNCATEDS: self.truncateds[transitions_to_save],
                    c.NEXT_OBSERVATIONS: self.next_observations[transitions_to_save],
                    c.NEXT_HIDDEN_STATES: self.next_hidden_states[transitions_to_save],
                    c.INFOS: {
                        info_name: info_value[transitions_to_save]
                        for info_name, info_value in self.infos.items()
                    },
                },
                f,
            )

        self._checkpoint_idxes[transitions_to_save] = 1
        self._checkpoint_count += 1

    def push(
        self,
        obs: np.ndarray,
        h_state: np.ndarray,
        act: np.ndarray,
        rew: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        next_obs: np.ndarray,
        next_h_state: np.ndarray,
        **kwargs,
    ) -> bool:
        self.next_observations[self._pointer] = next_obs
        self.next_hidden_states[self._pointer] = next_h_state

        return super().push(
            obs=obs,
            h_state=h_state,
            act=act,
            rew=rew,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def get_next(self, next_idxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        next_obss = self.next_observations[next_idxes]
        next_h_states = self.next_hidden_states[next_idxes]
        return next_obss, next_h_states

    def get_buffer_dict(self) -> Dict[str, Any]:
        buffer_dict = super().get_buffer_dict()
        buffer_dict[c.NEXT_OBSERVATIONS] = self.next_observations
        buffer_dict[c.NEXT_HIDDEN_STATES] = self.next_hidden_states
        return buffer_dict

    def load_from_buffer_dict(self, buffer_dict: Dict[str, Any]):
        super().load_from_buffer_dict(buffer_dict)
        self.next_observations = buffer_dict[c.NEXT_OBSERVATIONS]
        self.next_hidden_states = buffer_dict[c.NEXT_HIDDEN_STATES]


class TrajectoryNumPyBuffer(AbstractNumPyBuffer):
    """This buffer stores one trajectory as a sample"""

    def __init__(
        self,
        buffer_size: int,
        obs_dim: Sequence[int],
        h_state_dim: Sequence[int],
        act_dim: Sequence[int],
        rew_dim: Sequence[int],
        infos: dict = dict(),
        burn_in_window: int = 0,
        padding_first: bool = False,
        checkpoint_interval: int = 0,
        checkpoint_path: Optional[str] = None,
        rng: np.random.RandomState = np.random.RandomState(),
        dtype: np.dtype = np.float32,
        load_buffer: str = None,
    ):
        self._curr_episode_length = 0
        self._episode_lengths = [0]
        self._episode_start_idxes = [0]
        self._last_observations = [np.zeros(obs_dim)]
        self._last_h_states = [np.zeros(h_state_dim)]
        super().__init__(
            buffer_size=buffer_size,
            obs_dim=obs_dim,
            h_state_dim=h_state_dim,
            act_dim=act_dim,
            rew_dim=rew_dim,
            infos=infos,
            burn_in_window=burn_in_window,
            padding_first=padding_first,
            checkpoint_interval=checkpoint_interval,
            checkpoint_path=checkpoint_path,
            rng=rng,
            dtype=dtype,
            load_buffer=load_buffer,
        )

    def push(
        self,
        obs: np.ndarray,
        h_state: np.ndarray,
        act: np.ndarray,
        rew: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        next_obs: np.ndarray,
        next_h_state: np.ndarray,
        **kwargs,
    ) -> bool:
        self._episode_lengths[-1] += 1
        self._last_observations[-1] = next_obs
        self._last_h_states[-1] = next_h_state
        if self.is_full:
            self._episode_lengths[0] -= 1
            self._episode_start_idxes[0] += 1
            if self._episode_lengths[0] <= 0:
                self._episode_lengths.pop(0)
                self._episode_start_idxes.pop(0)
                self._last_observations.pop(0)
                self._last_h_states.pop(0)
        if terminated or truncated:
            self._episode_lengths.append(0)
            self._episode_start_idxes.append(self._pointer + 1)
            self._last_observations.append(np.zeros(obs.shape))
            self._last_h_states.append(np.zeros(h_state.shape))

        return super().push(
            obs=obs,
            h_state=h_state,
            act=act,
            rew=rew,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def sample(
        self,
        batch_size: int,
        idxes: Optional[np.ndarray] = None,
        horizon_length: int = 2,
        **kwargs,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        assert (
            horizon_length >= 2
        ), f"horizon_length must be at least length of 2. Got: {horizon_length}"
        if not len(self._episode_lengths):
            raise NoSampleError

        if idxes is None:
            episode_idxes = self.rng.randint(
                int(self._episode_lengths[0] <= 1),
                len(self._episode_lengths) - int(self._episode_lengths[-1] <= 1),
                size=batch_size,
            )
        else:
            episode_idxes = idxes

        # Get subtrajectory within each episode
        episode_lengths = np.array(self._episode_lengths)
        episode_start_idxes = np.array(self._episode_start_idxes)
        batch_episode_lengths = episode_lengths[episode_idxes]

        sample_lengths = np.tile(np.arange(horizon_length), (batch_size, 1))
        subtraj_start_idxes = self.rng.randint(batch_episode_lengths - 1)
        sample_idxes = (
            (subtraj_start_idxes + episode_start_idxes[episode_idxes])[:, None]
            + sample_lengths
        ) % self._buffer_size
        (
            obss,
            h_states,
            acts,
            rews,
            dones,
            terminateds,
            truncateds,
            infos,
            lengths,
        ) = self.get_transitions(sample_idxes.reshape(-1))

        traj_lengths = np.clip(
            batch_episode_lengths - subtraj_start_idxes, a_min=0, a_max=horizon_length
        ).astype(np.int64)
        sample_mask = np.flip(
            np.cumsum(np.eye(horizon_length)[horizon_length - traj_lengths], axis=-1),
            axis=-1,
        )
        sample_idxes = sample_idxes * sample_mask - np.ones(sample_idxes.shape) * (
            1 - sample_mask
        )
        infos[c.EPISODE_IDXES] = episode_idxes

        # If the episode ends too early, then the last observation should be in the trajectory
        # at index length_i of the trajectory.
        for sample_i, (ep_i, length_i) in enumerate(zip(episode_idxes, traj_lengths)):
            if length_i == horizon_length:
                continue
            obss[sample_i * horizon_length + length_i] = self._last_observations[ep_i]
            h_states[sample_i * horizon_length + length_i] = self._last_h_states[ep_i]

        return (
            obss,
            h_states,
            acts,
            rews,
            dones,
            terminateds,
            truncateds,
            infos,
            lengths,
            traj_lengths,
            sample_idxes,
        )

    def get_buffer_dict(self) -> Dict[str, Any]:
        buffer_dict = super().get_buffer_dict()
        buffer_dict[c.LAST_OBSERVATIONS] = self._last_observations
        buffer_dict[c.LAST_HIDDEN_STATES] = self._last_h_states
        buffer_dict[c.EPISODE_LENGTHS] = self._episode_lengths
        buffer_dict[c.EPISODE_START_IDXES] = self._episode_start_idxes
        buffer_dict[c.CURR_EPISODE_LENGTH] = self._curr_episode_length
        return buffer_dict

    def load_from_buffer_dict(self, buffer_dict: Dict[str, Any]):
        super().load_from_buffer_dict(buffer_dict)
        self._curr_episode_length = buffer_dict[c.CURR_EPISODE_LENGTH]
        self._last_observations = buffer_dict[c.LAST_OBSERVATIONS]
        self._last_h_states = buffer_dict[c.LAST_HIDDEN_STATES]
        self._episode_lengths = buffer_dict[c.EPISODE_LENGTHS]
        self._episode_start_idxes = buffer_dict[c.EPISODE_START_IDXES]
