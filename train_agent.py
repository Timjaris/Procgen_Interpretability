# import os
import csv
import gym
from stable_baselines3 import PPO

class LoggingWrapper(gym.Wrapper):
    def __init__(self, env, csv_path="episode_logs.csv"):
        super(LoggingWrapper, self).__init__(env)
        self.env = env
        self.ep_reward = 0
        self.ep_len = 0
        self.ep_rewards = []
        self.ep_lens = []
        self.csv_path = csv_path
        
        # Always write a new header line to the CSV file, regardless of existence
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode Length", "Total Reward"])  # Header for a new run


    
    def step(self, action):
        # Take the step in the environment
        observation, reward, done, info = self.env.step(action)
        
        self.ep_reward += reward
        self.ep_len += 1
        
        # log_pairs(observation)

        return observation, reward, done, info
    
    def reset(self, **kwargs):
        # Write the episode length and total reward to the CSV file when the episode ends
        if self.ep_len > 0:  # Ensure not to log the first reset
            with open(self.csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.ep_len, self.ep_reward])

        # Append rewards and lengths to lists
        self.ep_rewards.append(self.ep_reward)
        self.ep_lens.append(self.ep_len)

        # Reset episode stats
        self.ep_reward = 0
        self.ep_len = 0
        
        observation = self.env.reset(**kwargs)
        return observation
    


# env = gym.make('procgen:procgen-bossfight-v0', render_mode="human", num_levels=128, use_backgrounds=False)
env = gym.make('procgen:procgen-bossfight-v0', num_levels=128, use_backgrounds=False)

env = LoggingWrapper(env)

model = PPO("CnnPolicy", env)
# model.load("basic_cnn_mid_training7")
# model.learn(1000000)
model.save("untrained_cnn")
# env.close()



