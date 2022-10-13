from typing import Sequence, Callable, Any, Dict

import _pickle as pickle
import equinox as eqx
import numpy as np
import os
import timeit
import wandb

import jax_learning.wandb_constants as w


def compose(transforms: Sequence[Callable], input: Any, *args, **kwargs) -> Any:
    result = input
    for transform in transforms:
        result = transform(result, *args, **kwargs)
    return result


def init_wandb(**kwargs):
    wandb.init(**kwargs)
    wandb.define_metric(w.EPISODE_LENGTH, step_metric=w.EPISODE)
    wandb.define_metric(w.EPISODIC_RETURN, step_metric=w.EPISODE)


def polyak_average_generator(
    x: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def polyak_average(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return x * p + (1 - x) * q

    return polyak_average


def save_checkpoint(save_path: str, file_prefix: str, data: Dict[str, Any]):
    base_path = os.path.join(save_path, file_prefix)
    os.makedirs(base_path, exist_ok=True)

    pkl_data = {}
    for key, val in data.items():
        if isinstance(val, eqx.Module):
            eqx.tree_serialise_leaves(os.path.join(base_path, f"{key}-chkpt.eqx"), val)
        elif isinstance(val, dict):
            save_checkpoint(base_path, key, val)
        else:
            pkl_data[key] = val
    if len(pkl_data):
        pickle.dump(
            pkl_data, open(os.path.join(base_path, f"{file_prefix}-chkpt.pkl"), "wb")
        )


def load_checkpoint(load_path: str) -> Dict[str, Any]:
    data = {}
    for filename in os.listdir(load_path):
        complete_filename = os.path.join(load_path, filename)
        print(complete_filename)
        if os.path.isdir(complete_filename):
            data[filename] = load_checkpoint(complete_filename)
        else:
            key = os.path.basename(filename).split("-chkpt")[0]
            ext = os.path.basename(filename).split(".")[-1]
            if ext == "pkl":
                data.update(pickle.load(open(complete_filename, "rb")))
            elif ext == "eqx":
                data[key] = complete_filename
            else:
                print(f"Unsupported file extension: {ext} for file {complete_filename}")
                raise NotImplementedError
    return data


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


CONTENT = "content"
LOG_SETTING = "log_setting"
STANDARD_DEVIATION = "standard_deviation"
MIN_MAX = "min_max"
AXIS = "axis"


class EpochSummary:
    def __init__(self, name, default_key_length: int = 10, padding: int = 11):
        self._name = name
        self._key_length = default_key_length
        self._padding = padding
        self._summary = dict()
        self._epoch = 0
        self._init_tic = timeit.default_timer()

    def log(
        self,
        key: str,
        value: Any,
        track_std: bool = True,
        track_min_max: bool = True,
        axis: int = None,
    ):
        self._key_length = max(self._key_length, len(key))
        self._summary.setdefault(
            key,
            {
                LOG_SETTING: {
                    STANDARD_DEVIATION: track_std,
                    MIN_MAX: track_min_max,
                    AXIS: axis,
                },
                CONTENT: [],
            },
        )
        self._summary[key][CONTENT].append(value)

    def new_epoch(self):
        self._epoch += 1
        self._summary.clear()
        self._curr_tic = timeit.default_timer()

    def print_summary(self):
        toc = timeit.default_timer()
        key_length = self._key_length + self._padding
        print("=" * 100)
        print(f"{self._name}")
        print(f"Epoch: {self._epoch}")
        print(f"Epoch Time Spent: {toc - self._curr_tic}")
        print(f"Total Time Spent: {toc - self._init_tic}")
        print("=" * 100)
        print("|".join(str(x).ljust(key_length) for x in ("Key", "Content")))
        print("-" * 100)
        for key in sorted(self._summary):
            val = self._summary[key][CONTENT]
            setting = self._summary[key][LOG_SETTING]
            try:
                print(
                    "|".join(
                        str(x).ljust(key_length)
                        for x in (f"{key} - AVG", np.mean(val, axis=setting[AXIS]))
                    )
                )
                if setting[STANDARD_DEVIATION]:
                    print(
                        "|".join(
                            str(x).ljust(key_length)
                            for x in (
                                f"{key} - STD DEV",
                                np.std(val, axis=setting[AXIS]),
                            )
                        )
                    )
                if setting[MIN_MAX]:
                    print(
                        "|".join(
                            str(x).ljust(key_length)
                            for x in (f"{key} - MIN", np.min(val, axis=setting[AXIS]))
                        )
                    )
                    print(
                        "|".join(
                            str(x).ljust(key_length)
                            for x in (f"{key} - MAX", np.max(val, axis=setting[AXIS]))
                        )
                    )
            except:
                print(val)
                print(key)
                assert 0
        print("=" * 100)
