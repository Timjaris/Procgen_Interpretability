import numpy as np
import csv
import matplotlib.pyplot as plt

# Initialize lists to store episode lengths and total rewards
episode_lengths = []
total_rewards = []
total_timesteps = 0
# Open the CSV file and read the data
with open('episode_logs_fragment_model_1-7.csv', mode='r') as file:  # Replace with your actual file path
    reader = csv.reader(file)
    next(reader)  # Skip the header
    i=0
    for row in reader:
        # print(i)
        i+=1
        try:
            # Convert data to appropriate types and append to lists
            total_timesteps += int(row[0])
            episode_lengths.append(int(row[0]))
            total_rewards.append(float(row[1]))
        except:
            pass
        # if (float(row[1])>10):
        #     print("HUH?!", i)

print("total timesteps =", total_timesteps)
# # Plotting the reward over time
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(total_rewards) + 1), total_rewards, marker='o', linestyle='-', color='b', alpha=0.5)
# plt.xlabel("Episode")
# plt.ylabel("Total Reward")
# plt.title("Reward Over Time")
# plt.grid(True)

# # Set y-axis limits based on the min and max of your rewards
# plt.ylim([min(total_rewards) - 0.5, max(total_rewards) + 0.5])

# # Improve the y-axis formatting: fewer ticks and better formatting
# plt.gca().yaxis.set_major_locator(plt.MaxNLocator(10))
# plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

# plt.show()


# Calculate the average reward and standard deviation for every 100 episodes
chunk_size = 2000
average_rewards = [np.mean(total_rewards[i:i + chunk_size]) for i in range(0, len(total_rewards), chunk_size)]
std_devs = [np.std(total_rewards[i:i + chunk_size]) for i in range(0, len(total_rewards), chunk_size)]

# Plotting the average reward over time with standard deviations
plt.figure(figsize=(10, 6))
plt.errorbar(range(1, len(average_rewards) + 1), average_rewards, yerr=std_devs, fmt='o-', color='b', alpha=0.7, ecolor='r', capsize=3)
plt.xlabel("Chunk of 100 Episodes")
plt.ylabel("Average Total Reward")
plt.title("Average Reward Over 100 Episode Chunks with Standard Deviation")
plt.grid(True)

plt.show()