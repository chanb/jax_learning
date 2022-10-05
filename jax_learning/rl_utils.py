import _pickle as pickle
import copy
import os
import numpy as np
import sys
import timeit
import wandb

from argparse import Namespace
from datetime import datetime
from typing import Sequence, Callable, Any, Union

import jax_learning.constants as c
import jax_learning.wandb_constants as w

from jax_learning.agents import Agent
from jax_learning.common import EpochSummary


def random_exploration_generator(
    exploration_strategy: str,
    action_dim: Sequence[int],
    min_action: Union[float, np.ndarray] = -1.0,
    max_action: Union[float, np.ndarray] = 1.0,
) -> Callable[[], np.ndarray]:
    def sample_standard_gaussian():
        return np.clip(np.random.randn(*action_dim), a_min=min_action, a_max=max_action)

    def sample_uniform_categorical():
        return np.random.randint(0, action_dim[0])

    if exploration_strategy == c.STANDARD_GAUSSIAN:
        return sample_standard_gaussian
    elif exploration_strategy == c.UNIFORM_CATEGORICAL:
        return sample_uniform_categorical
    else:
        raise NotImplementedError


def interact(env: Any, agent: Agent, cfg: Namespace):
    max_timesteps = cfg.max_timesteps
    log_interval = cfg.log_interval
    render = env.render if cfg.render else lambda: None
    env_rng = cfg.env_rng
    evaluation_frequency = cfg.evaluation_frequency
    evaluation_env = copy.deepcopy(env)
    checkpoint_frequency = cfg.checkpoint_frequency
    save_path = cfg.save_path

    if checkpoint_frequency:
        time_tag = datetime.strftime(datetime.now(), "%m-%d-%y_%H_%M_%S")
        if save_path is None:
            save_path = "unnamed_run"
        save_path = f"{save_path}/{time_tag}"
        checkpoint_path = os.path.join(save_path, "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)

    random_exploration = getattr(cfg, c.RANDOM_EXPLORATION, None)
    num_exploration = getattr(cfg, c.EXPLORATION_STEPS, 0)

    clip_action = getattr(cfg, c.CLIP_ACTION, False)
    if clip_action:
        max_action = cfg.max_action
        min_action = cfg.min_action

    obs, info = env.reset(seed=env_rng.randint(0, sys.maxsize))
    h_state = agent.reset()
    ep_return = 0.0
    ep_i = 0
    ep_len = 0
    metrics_batch = []
    epoch_summary = EpochSummary()
    epoch_summary.new_epoch()
    tic = timeit.default_timer()
    for timestep_i in range(max_timesteps):
        timestep_dict = {f"{w.TRAIN}/{w.TIMESTEP}": timestep_i}

        act, next_h_state = agent.compute_action(obs, h_state, timestep_dict)

        if timestep_i < num_exploration:
            act = random_exploration()

        env_act = act
        if clip_action:
            env_act = np.clip(act, min_action, max_action)
        next_obs, rew, terminated, truncated, info = env.step(env_act)
        render()
        agent.store(
            obs, h_state, act, rew, terminated, truncated, info, next_obs, next_h_state
        )
        agent.learn(next_obs, next_h_state, timestep_dict, epoch_summary)

        obs = next_obs
        ep_return += rew
        ep_len += 1

        if terminated or truncated:
            obs, info = env.reset(seed=env_rng.randint(0, sys.maxsize))
            h_state = agent.reset()
            epoch_summary.log(f"{w.TRAIN}/{w.EPISODIC_RETURN}", ep_return, axis=0)
            epoch_summary.log(f"{w.TRAIN}/{w.EPISODE_LENGTH}", ep_len)
            timestep_dict[f"{w.TRAIN}/{w.EPISODIC_RETURN}"] = ep_return
            timestep_dict[f"{w.TRAIN}/{w.EPISODE}"] = ep_i
            timestep_dict[f"{w.TRAIN}/{w.EPISODE_LENGTH}"] = ep_len
            ep_return = 0.0
            ep_len = 0
            ep_i += 1

        if evaluation_frequency and (timestep_i + 1) % evaluation_frequency == 0:
            evaluate(
                evaluation_env, agent, cfg.evaluation_cfg, timestep_dict, epoch_summary
            )

        if checkpoint_frequency and (timestep_i + 1) % checkpoint_frequency == 0:
            agent_dict = agent.checkpoint()
            pickle.dump(open(checkpoint_path, "wb"), agent_dict)

        metrics_batch.append(timestep_dict)
        if (timestep_i + 1) % log_interval == 0:
            epoch_summary.print_summary()
            epoch_summary.new_epoch()

            toc = timeit.default_timer()
            timestep_dict[c.TIMEDIFF] = toc - tic
            tic = timeit.default_timer()

            for metrics in metrics_batch:
                wandb.log(metrics)
            metrics_batch = []

    for metrics in metrics_batch:
        wandb.log(metrics)


def evaluate(
    env: Any,
    agent: Agent,
    cfg: Namespace,
    timestep_dict: dict,
    epoch_summary: EpochSummary,
):
    num_episodes = cfg.num_episodes
    render = env.render if cfg.render else lambda: None
    env_rng = cfg.env_rng
    clip_action = getattr(cfg, c.CLIP_ACTION, False)
    if clip_action:
        max_action = cfg.max_action
        min_action = cfg.min_action

    timestep_i = 0
    ep_lengths = []
    ep_returns = []
    for _ in range(num_episodes):
        obs, info = env.reset(seed=env_rng.randint(0, sys.maxsize))
        h_state = agent.reset()
        ep_return = 0.0
        ep_length = 0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            act, h_state = agent.deterministic_action(obs, h_state, timestep_dict)
            env_act = act
            if clip_action:
                env_act = np.clip(act, min_action, max_action)
            obs, rew, terminated, truncated, info = env.step(env_act)
            render()
            ep_return += rew
            ep_length += 1
            timestep_i += 1

            if terminated or truncated:
                ep_returns.append(ep_return)
                ep_lengths.append(ep_length)
    timestep_dict[f"{w.EVALUATION}/mean_{c.EPISODIC_RETURN}"] = np.mean(ep_returns)
    timestep_dict[f"{w.EVALUATION}/mean_{c.EPISODE_LENGTH}"] = np.mean(ep_lengths)

    epoch_summary.log(
        f"{w.EVALUATION}/mean_{c.EPISODIC_RETURN}", np.mean(ep_returns), axis=0
    )
    epoch_summary.log(f"{w.EVALUATION}/mean_{c.EPISODE_LENGTH}", np.mean(ep_lengths))
