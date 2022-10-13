import gym
import d4rl  # Import required to register environments
import d4rl_atari

"""
Install from:
- https://github.com/Farama-Foundation/D4RL
- https://github.com/takuseno/d4rl-atari
"""

# env_name = "hopper-expert-v0"
# env_name = "breakout-mixed-v0"
env_name = "minigrid-fourrooms-v0"

type = "d4rl"
# type = "d4rl_atari"
valid_types = ("d4rl", "d4rl_atari")

assert type in valid_types, f"{type} is not in a valid environment type: {valid_types}"

# Create the environment
env = gym.make(env_name)

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

if type == "d4rl":
    # Alternatively, use d4rl.qlearning_dataset which
    # also adds next_observations.
    dataset = d4rl.qlearning_dataset(env)
elif type == "d4rl_atari":
    dataset = env.get_dataset()
print(dataset.keys())  # An N x dim_observation Numpy array of observations
