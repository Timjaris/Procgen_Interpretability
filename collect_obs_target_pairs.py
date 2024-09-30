import re

import gym
import time
import pickle
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split



# register_env("wrapper-v0", env_creator)
def read_vector_from_file(file_path):
    with open(file_path, 'r') as file:
        # Read the line containing the vector
        line = file.readline().strip()
        
        # Strip the parentheses
        line = line.strip('()')
        
        vector = [float(item) for item in re.split(r'[;,]', line) if item]
    
    return vector

def get_closest(agent_pos, other_poses):
    # Convert agent_pos and other_poses to numpy arrays
    agent_pos = np.array(agent_pos)
    other_poses = np.array(other_poses).reshape(-1, 2)  # Reshape other_poses to (n, 2)
    
    # Compute the Euclidean distance between agent_pos and all points in other_poses
    distances = np.linalg.norm(other_poses - agent_pos, axis=1)
    
    # Find the index of the minimum distance
    try:
        closest_idx = np.argmin(distances)
    except ValueError:
        return -1,-1
    
    # Return the closest position
    return tuple(other_poses[closest_idx])



def collect_pairs(run_name, train_or_test, run_id="0"):
    print("Collecting Pairs")
    start = time.time()
    env = gym.make('procgen:procgen-bossfight-v0', num_levels=128, use_backgrounds=False)

    # log_dir = f"/results/{run_name}/"
    model = PPO("CnnPolicy", env)
    model.load("untrained_cnn")
    
    # #todo, mess around with making these numbers bigger
    timesteps = 100000
    if train_or_test == "test": timesteps //= 4
    obses = []
    agent_poses = []
    boss_poses = []
    bullet_poses = []
    rock_poses = []
    
    obs = env.reset()
    start = time.time()
    for i in range(timesteps):
        if i!=0 and (i % (timesteps // 100) == 0 or i == timesteps - 1):
            print(100 * i / timesteps, "%", "Runtime:", time.time()-start, "Projected Runtime:", (timesteps-i)/(i/(time.time() - start)))
        # time.sleep(.01) #TODO: verify reading vector from file still works <0.1 or without it
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        agent_pos = read_vector_from_file("agent_position.txt")
        boss_pos = read_vector_from_file("boss_position.txt")
        #Using linear probes to predict and arbitrary list sounds difficult, so 
        #I'm just having the agent predict the closest one. That's the one more 
        #important anyway.
        #Well, it could be hiding behind one rock, while being closer to another, but that's an edge case.
        obses.append(obs)
        bullet_pos = get_closest(agent_pos, read_vector_from_file("bullet_positions.txt"))
        rock_pos = get_closest(agent_pos, read_vector_from_file("rock_positions.txt"))
        agent_poses.append(agent_pos)
        boss_poses.append(boss_pos)
        bullet_poses.append(bullet_pos)
        rock_poses.append(rock_pos)
        
        
        # env.render()
        if done:
          obs = env.reset()
    env.close()
          
    save_path_train = f"data/{run_name}_{train_or_test}_data.pkl"
    # Save the data using pickle
    with open(save_path_train, 'wb') as f:
        pickle.dump({'obses': obses, 'agent_poses': agent_poses,'boss_poses': 
                     boss_poses, 'bullet_poses':bullet_poses, 'rock_poses':
                     rock_poses}, f)
        
    print("Pair collection Runtime:", time.time() - start)


if __name__ == "__main__":
    collect_pairs("untrained", "train")
    # collect_pairs("delete_these", "train")
    # collect_pairs("pairs", "test")
