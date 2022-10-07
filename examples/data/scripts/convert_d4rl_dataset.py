import h5py
import numpy as np
import os

from tqdm import tqdm

from jax_learning.buffers.ram_buffers import NextStateNumPyBuffer

base_h5path = "/Users/chanb/.d4rl/datasets"
env = "hopper_medium_expert-v2"
save_path = f"./{env}.pkl"

OBSERVATIONS = "observations"
NEXT_OBSERVATIONS = "next_observations"
ACTIONS = "actions"
REWARDS = "rewards"
TERMINALS = "terminals"
TIMEOUTS = "timeouts"


# Code from D4RL
def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


data_dict = {}
with h5py.File(os.path.join(base_h5path, f"{env}.hdf5"), "r") as dataset_file:
    for k in tqdm(get_keys(dataset_file), desc="load datafile"):
        try:  # first try loading as an array
            data_dict[k] = dataset_file[k][:]
        except ValueError as e:  # try loading as a scalar
            data_dict[k] = dataset_file[k][()]
print(data_dict)

buffer_size = len(data_dict[OBSERVATIONS])
obs_dim = data_dict[OBSERVATIONS].shape[1:]
h_state_dim = (1,)
act_dim = data_dict[ACTIONS].shape[1:]
rew_dim = (1,)
buffer_rng = np.random.RandomState(0)

buffer = NextStateNumPyBuffer(
    buffer_size=buffer_size,
    obs_dim=obs_dim,
    h_state_dim=h_state_dim,
    act_dim=act_dim,
    rew_dim=rew_dim,
    rng=buffer_rng,
)

h_state = np.array([0.0])
for (obs, act, rew, terminal, timeout, next_obs) in tqdm(
    zip(
        data_dict[OBSERVATIONS],
        data_dict[ACTIONS],
        data_dict[REWARDS],
        data_dict[TERMINALS],
        data_dict[TIMEOUTS],
        data_dict[NEXT_OBSERVATIONS],
    ),
    desc="push data",
):
    buffer.push(
        obs=obs,
        h_state=h_state,
        act=act,
        rew=rew,
        terminated=terminal,
        truncated=timeout,
        info={},
        next_obs=next_obs,
        next_h_state=h_state,
    )
buffer.save(save_path=save_path, end_with_done=False)
