import sys
import timeit
import wandb

import numpy as np

import jax_learning.constants as c
import jax_learning.wandb_constants as w


def interact(env, agent, cfg):
    max_timesteps = cfg.max_timesteps
    log_interval = cfg.log_interval
    render = env.render if cfg.render else lambda: None
    env_rng = cfg.env_rng
    clip_action = getattr(cfg, c.CLIP_ACTION, False)
    if clip_action:
        max_action = cfg.max_action
        min_action = cfg.min_action

    obs = env.reset(seed=env_rng.randint(0, sys.maxsize))
    h_state = agent.reset()
    ep_return = 0.0
    ep_i = 0
    ep_len = 0
    metrics_batch = []
    tic = timeit.default_timer()
    for timestep_i in range(max_timesteps):
        timestep_dict = {w.TRAIN: {w.TIMESTEP: timestep_i}}
        act, next_h_state = agent.compute_action(obs, h_state, timestep_dict)
        env_act = act
        if clip_action:
            env_act = np.clip(act, min_action, max_action)
        next_obs, rew, done, info = env.step(env_act)
        render()
        agent.store(obs, h_state, act, rew, done, info, next_obs, next_h_state)
        agent.learn(next_obs, next_h_state, timestep_dict)
        obs = next_obs
        ep_return += rew
        ep_len += 1

        if done:
            obs = env.reset(seed=env_rng.randint(0, sys.maxsize))
            h_state = agent.reset()
            timestep_dict[w.TRAIN][w.EPISODIC_RETURN] = ep_return
            timestep_dict[w.TRAIN][w.EPISODE] = ep_i
            timestep_dict[w.TRAIN][w.EPISODE_LENGTH] = ep_len
            ep_return = 0.0
            ep_len = 0
            ep_i += 1

        metrics_batch.append(timestep_dict)
        if (timestep_i + 1) % log_interval == 0:
            toc = timeit.default_timer()
            timestep_dict[c.TIMEDIFF] = toc - tic
            tic = timeit.default_timer()
            for metrics in metrics_batch:
                wandb.log(metrics)
            metrics_batch = []

    for metrics in metrics_batch:
        wandb.log(metrics)


def evaluate(env, agent, cfg):
    max_episodes = cfg.max_episodes
    render = env.render if cfg.render else lambda: None
    env_rng = cfg.env_rng
    clip_action = getattr(cfg, c.CLIP_ACTION, False)
    if clip_action:
        max_action = cfg.max_action
        min_action = cfg.min_action

    timestep_i = 0
    metrics_batch = []
    for ep_i in range(max_episodes):
        obs = env.reset(seed=env_rng.randint(0, sys.maxsize))
        h_state = agent.reset()
        ep_return = 0.0
        ep_len = 0
        done = False
        while not done:
            timestep_dict = {w.EVALUATION: {w.TIMESTEP: timestep_i}}
            act, h_state = agent.deterministic_action(obs, h_state, timestep_dict)
            env_act = act
            if clip_action:
                env_act = np.clip(act, min_action, max_action)
            obs, rew, done, info = env.step(env_act)
            render()
            ep_return += rew
            ep_len += 1
            timestep_i += 1

            if done:
                timestep_dict[w.EVALUATION][w.EPISODIC_RETURN] = ep_return
                timestep_dict[w.EVALUATION][w.EPISODE] = ep_i
                timestep_dict[w.EVALUATION][w.EPISODE_LENGTH] = ep_len

            metrics_batch.append(timestep_dict)
    for metrics in metrics_batch:
        wandb.log(metrics)
