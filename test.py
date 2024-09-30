import gym
import procgen

env = gym.make('procgen:procgen-bossfight-v0', render_mode="human")
env.reset()
env.step(0)