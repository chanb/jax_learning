import gym
import d4rl  # Import required to register environments

# Create the environment
env = gym.make("hopper-medium-expert-v2")

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
dataset = d4rl.qlearning_dataset(env)
print(dataset["observations"])  # An N x dim_observation Numpy array of observations
