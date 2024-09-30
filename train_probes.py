import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import os
import gym
import csv
import time
import pickle
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
    

def reconstruct_model(obs):
    obs = torch.tensor(np.array([obs]), dtype=torch.float32)

    # Rearrange the dimensions to match PyTorch's expectation [batch_size, channels, height, width]
    obs = obs.permute(0, 3, 1, 2)  # Convert from [batch_size, height, width, channels] to [batch_size, channels, height, width]

    # Get model weights and biases
    layers = model.policy.state_dict()

    # Extract weights and biases for the CNN layers
    conv1_w = layers["features_extractor.cnn.0.weight"]
    conv1_b = layers["features_extractor.cnn.0.bias"]
    conv2_w = layers["features_extractor.cnn.2.weight"]
    conv2_b = layers["features_extractor.cnn.2.bias"]
    conv3_w = layers["features_extractor.cnn.4.weight"]
    conv3_b = layers["features_extractor.cnn.4.bias"]

    # Extract weights and biases for the linear layer in features_extractor
    linear_w = layers["features_extractor.linear.0.weight"]
    linear_b = layers["features_extractor.linear.0.bias"]

    # Process input through the CNN
    x = F.conv2d(obs, conv1_w, conv1_b, stride=4)
    x = F.relu(x)
    # conv1_out = x  # Save the first convolution output
    
    x = F.conv2d(x, conv2_w, conv2_b, stride=2)
    x = F.relu(x)
    # conv2_out = x  # Save the second convolution output
    
    x = F.conv2d(x, conv3_w, conv3_b, stride=1)
    x = F.relu(x)
    # conv3_out = x  # Save the third convolution output
    
    # Flatten the CNN output
    x = x.reshape(x.size(0), -1)  # Use reshape instead of view

    # The output from the CNN should have 1024 features, as expected
    assert x.shape[1] == 1024, f"Expected CNN output to have 1024 features, got {x.shape[1]}"

    # Pass through policy feature extractor (before reducing dimensionality)
    pi_linear_w = layers["pi_features_extractor.linear.0.weight"]
    pi_linear_b = layers["pi_features_extractor.linear.0.bias"]
    pi_out = F.relu(torch.matmul(x, pi_linear_w.T) + pi_linear_b)  # Transpose the weight matrix

    # Pass through value feature extractor (before reducing dimensionality)
    vf_linear_w = layers["vf_features_extractor.linear.0.weight"]
    vf_linear_b = layers["vf_features_extractor.linear.0.bias"]
    vf_out = F.relu(torch.matmul(x, vf_linear_w.T) + vf_linear_b)  # Transpose the weight matrix

    # Now, reduce the 1024-dimensional feature vector to 512 for the shared linear layer
    # linear_out = F.relu(torch.matmul(x, linear_w.T) + linear_b)

    # Extract action and value network layers
    action_w = layers["action_net.weight"]
    action_b = layers["action_net.bias"]
    value_w = layers["value_net.weight"]
    value_b = layers["value_net.bias"]

    # Compute action logits and state value
    action = torch.matmul(pi_out, action_w.T) + action_b
    value = torch.matmul(vf_out, value_w.T) + value_b
    
    return action, value


def train_probes_batch(model, batched_obs, batched_targets, probes, optimizers, loss_fn, inference=False):
    # Convert observations and targets to tensors
    obs = torch.tensor(np.array(batched_obs), dtype=torch.float32)
    targets = torch.tensor(np.array(batched_targets), dtype=torch.float32)

    # Rearrange the dimensions to match PyTorch's expectation [batch_size, channels, height, width]
    obs = obs.permute(0, 3, 1, 2)  # Convert from [batch_size, height, width, channels] to [batch_size, channels, height, width]

    obs_flat = obs.reshape(obs.size(0), -1)  # This will flatten the obs to [batch_size, 12288]

    # Get model weights and biases
    layers = model.policy.state_dict()

    # Extract weights and biases for the CNN layers
    conv1_w = layers["features_extractor.cnn.0.weight"]
    conv1_b = layers["features_extractor.cnn.0.bias"]
    conv2_w = layers["features_extractor.cnn.2.weight"]
    conv2_b = layers["features_extractor.cnn.2.bias"]
    conv3_w = layers["features_extractor.cnn.4.weight"]
    conv3_b = layers["features_extractor.cnn.4.bias"]

    # Extract weights and biases for the linear layer in features_extractor
    linear_w = layers["features_extractor.linear.0.weight"]
    linear_b = layers["features_extractor.linear.0.bias"]

    # Process input through the CNN
    x = F.conv2d(obs, conv1_w, conv1_b, stride=4)
    x = F.relu(x)
    conv1_out = x  # Save the first convolution output
    conv1_out_flat = conv1_out.reshape(conv1_out.size(0), -1)  # Flatten conv1 output
    
    x = F.conv2d(x, conv2_w, conv2_b, stride=2)
    x = F.relu(x)
    conv2_out = x  # Save the second convolution output
    conv2_out_flat = conv2_out.reshape(conv2_out.size(0), -1)  # Flatten conv2 output
    
    x = F.conv2d(x, conv3_w, conv3_b, stride=1)
    x = F.relu(x)
    conv3_out = x  # Save the third convolution output
    conv3_out_flat = conv3_out.reshape(conv3_out.size(0), -1)  # Flatten conv3 output

    
    # Flatten the CNN output
    x = x.reshape(x.size(0), -1)  # Use reshape instead of view

    # The output from the CNN should have 1024 features, as expected
    assert x.shape[1] == 1024, f"Expected CNN output to have 1024 features, got {x.shape[1]}"

    # Pass through policy feature extractor (before reducing dimensionality)
    pi_linear_w = layers["pi_features_extractor.linear.0.weight"]
    pi_linear_b = layers["pi_features_extractor.linear.0.bias"]
    pi_out = F.relu(torch.matmul(x, pi_linear_w.T) + pi_linear_b)  # Transpose the weight matrix

    # Pass through value feature extractor (before reducing dimensionality)
    vf_linear_w = layers["vf_features_extractor.linear.0.weight"]
    vf_linear_b = layers["vf_features_extractor.linear.0.bias"]
    vf_out = F.relu(torch.matmul(x, vf_linear_w.T) + vf_linear_b)  # Transpose the weight matrix

    # Now, reduce the 1024-dimensional feature vector to 512 for the shared linear layer
    linear_out = F.relu(torch.matmul(x, linear_w.T) + linear_b)

    # Extract action and value network layers
    action_w = layers["action_net.weight"]
    action_b = layers["action_net.bias"]
    value_w = layers["value_net.weight"]
    value_b = layers["value_net.bias"]

    # Compute action logits and state value
    action = torch.matmul(pi_out, action_w.T) + action_b
    value = torch.matmul(vf_out, value_w.T) + value_b

    # Collect activations for each layer
    layer_activations = [obs_flat, conv1_out_flat, conv2_out_flat, conv3_out_flat,
                         linear_out, pi_out, vf_out, action, value]
    names = ["obs", "conv1_out", "conv2_out", "conv3_out", "linear_out", 
             "pi_out", "vf_out", "action", "value"]

    # Train probes and apply optimizers
    losses = []
    for i, (probe, layer, name) in enumerate(zip(probes, layer_activations, names)):
        # print("probing", name)
        predicted = probe(layer)
        loss = loss_fn(predicted, targets)
        
        if not inference:
            optimizer = optimizers[i]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        losses.append(loss.item())

    return losses


# Create the linear probe as a PyTorch module
class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    start = time.time()
    
    
    run_folder = "run_1" 
    data_name = "pairs"
    # data_name = "delete_these"
    save_path_train = f"data/{data_name}_train_data.pkl" #manual
    save_path_test = "data/{data_name}_test_data.pkl" #manual
    
    env = gym.make('procgen:procgen-bossfight-v0', num_levels=128, use_backgrounds=False)
    obs = env.reset()

    # log_dir = f"/results/{run_name}/"
    model = PPO("CnnPolicy", env)
    model.load("basic_cnn_mid_training7")

    target_dim = 2

    # Define probes for each layer in layer_activations
    probe0 = LinearProbe(12288, target_dim)  # obs
    probe1 = LinearProbe(7200, target_dim)   # conv1_out
    probe2 = LinearProbe(2304, target_dim)   # conv2_out
    probe3 = LinearProbe(1024, target_dim)   # conv3_out
    probe4 = LinearProbe(512, target_dim)    # linear_out
    probe5 = LinearProbe(512, target_dim)    # pi_out (policy output)
    probe6 = LinearProbe(512, target_dim)    # vf_out (value function output)
    probe7 = LinearProbe(15, target_dim)     # action (action logits)
    probe8 = LinearProbe(1, target_dim)      # value (scalar value function)
    
    optimizer0 = optim.Adam(probe0.parameters(), lr=0.01)
    optimizer1 = optim.Adam(probe1.parameters(), lr=0.01)
    optimizer2 = optim.Adam(probe2.parameters(), lr=0.01)
    optimizer3 = optim.Adam(probe3.parameters(), lr=0.01)
    optimizer4 = optim.Adam(probe4.parameters(), lr=0.01)
    optimizer5 = optim.Adam(probe5.parameters(), lr=0.01)
    optimizer6 = optim.Adam(probe6.parameters(), lr=0.01)
    optimizer7 = optim.Adam(probe7.parameters(), lr=0.01)
    optimizer8 = optim.Adam(probe8.parameters(), lr=0.01)
    
    loss_fn = torch.nn.MSELoss()
    
    probes = [probe0, probe1, probe2, probe3, probe4, probe5, probe6, probe7, probe8]
    probe_names = ['obs','cnn1','cnn2','cnn3','linear','policy_net','value_net','action','value']
    probes_dict = {name: probe for name, probe in zip(probe_names, probes)}
    optimizers = [optimizer0, optimizer1, optimizer2, optimizer3, optimizer4,
                  optimizer5, optimizer6, optimizer7, optimizer8]    

        
    # Load the data using pickle
    with open(save_path_train, 'rb') as f:
        train_data = pickle.load(f)
        
    # with open(save_path_test, 'rb') as f:
    #     test_data = pickle.load(f)
    
    # Extract the lists
    train_obs = train_data['obses']
    agent_poses_train = train_data['agent_poses']
    boss_poses_train = train_data['boss_poses']
    bullet_poseses_train = train_data['bullet_poses']
    rock_poseses_train = train_data['rock_poses']
    
    # test_obs = test_data['obses']
    # agent_poses_test = test_data['agent_poses']
    # boss_poses_test = test_data['boss_poses']
    # bullet_poseses_test = test_data['bullet_poses']
    # rock_poseses_test = test_data['rock_poses']
    # print("loaded data")
    
    target_names = [ "Closest Bullet Position", "Closest Rock Position"]#, "Agent Position", "Boss Position",]
    # test_set_targets = agent_poses_test
    target_data = [bullet_poseses_train, rock_poseses_train]#, agent_poses_train, boss_poses_train]

    for target_name, train_set_targets in zip(target_names, target_data):
        
    
        epochs = 1000
        batch_size = 10000
        train_losses = {probe_name: [] for probe_name in probes_dict.keys()}
        test_losses = {probe_name: [] for probe_name in probes_dict.keys()}
        losseses, test_losseses = [], []
        
        best_losses = [float('inf')] * len(probes)
        epochs_without_improvement = 0
        max_epochs_without_improvement = 5  
        
        for epoch in range(epochs):
            improved = False
            if epoch != 0 and (epoch % (epochs // 100) == 0 or epoch == epochs - 1):
                if epoch != 0:
                    print("epoch", epoch, 100 * epoch / epochs, "%", "Runtime:", time.time()-start, "Projected Runtime:", (epochs-epoch)/(epoch/(time.time() - start)))
                    if (epoch % (epochs // 10)) == 0:
                        print("\tBest Losses per Probe:", best_losses)
                else: 
                    print("epoch", epoch)
                    
                    
            #Note: a train set doesn't make sense unless I'm partioning based on the 128 possible starting configs
            # test_losses = train_probes_batch(model, test_obs, test_set_targets, 
            #                             probes, optimizers, loss_fn, inference=True)
            # test_losseses.append(test_losses)
            
            for i in range(int(np.ceil((len(train_obs)/batch_size)))): 
                batched_obs = train_obs[i*batch_size:(i+1)*batch_size]
                batched_targets = train_set_targets[i*batch_size:(i+1)*batch_size]
                
                losses = train_probes_batch(model, batched_obs, batched_targets, 
                                            probes, optimizers, loss_fn)
                losseses.append(losses)
                
            # Check for improvements in probe losses
            for j in range(len(probes)):
                if losses[j] < best_losses[j]:
                    best_losses[j] = losses[j]
                    improved = True
                    
            # If no improvements in this epoch, increment the counter
            if not improved:
                epochs_without_improvement += 1
            else:
                epochs_without_improvement = 0
            
            # Break out of the loop if no improvement for 5 consecutive epochs
            if epochs_without_improvement >= max_epochs_without_improvement:
                print(f"Stopping early at epoch {epoch} due to lack of improvement.")
                break
    
        
        current_datetime = datetime.now().strftime("%Y-%m-%d_%p%I-%M")
        dated_run_folder = f"{target_name}_{current_datetime}"
        
        
        log_dir = "results"
        save_path = os.path.join(log_dir, dated_run_folder)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(probe0.state_dict(), os.path.join(save_path, "probe0.pth"))
        torch.save(probe1.state_dict(), os.path.join(save_path, "probe1.pth"))
        torch.save(probe2.state_dict(), os.path.join(save_path, "probe2.pth"))
        torch.save(probe3.state_dict(), os.path.join(save_path, "probe3.pth"))
        torch.save(probe4.state_dict(), os.path.join(save_path, "probe4.pth"))
        torch.save(probe5.state_dict(), os.path.join(save_path, "probe5.pth"))
        torch.save(probe6.state_dict(), os.path.join(save_path, "probe6.pth"))
        torch.save(probe7.state_dict(), os.path.join(save_path, "probe7.pth"))
        torch.save(probe8.state_dict(), os.path.join(save_path, "probe8.pth"))
        env.close()
        
        losses_per_probe = np.array(losseses).T
        # Generate a color map
        cmap = plt.get_cmap("tab10")
        
        # Plot training losses
        final_losses = []
        for i, probe_losses in enumerate(losses_per_probe):
            color = cmap(i)
            
            # Calculate the max value in the second half of the array
            second_half_max = np.max(probe_losses[len(probe_losses) // 2:])
            ylim_value = second_half_max * 2  # Cap the y-axis to double this value
            
            # Save individual loss curve with dynamic y-axis limit
            plt.figure()  # Create a new figure for each individual plot
            plt.plot(probe_losses, color=color, label=probe_names[i])
            plt.ylim(0, ylim_value)  # Cap the y-axis dynamically
            plt.xlabel('Batches')
            plt.ylabel('Loss')
            plt.title = f'Training Loss for {probe_names[i]}'  # Corrected title syntax
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_path, f'training_loss_{probe_names[i]}.png'), dpi=300, bbox_inches='tight')
            plt.clf()  # Clear the figure after saving
        
        # Save the combined loss curve plot with dynamic y-axis limit
        plt.figure()  # Create a new figure for the combined plot
        for i, probe_losses in enumerate(losses_per_probe):
            color = cmap(i)
            plt.plot(probe_losses, color=color, label=probe_names[i])
        
        # Set labels and title for the combined plot
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.title = 'Training and Test Losses for Different Probes'  # Corrected title syntax
        plt.legend()
        plt.grid(True)
        
        # Dynamically set y-axis limit for combined plot based on all probes
        combined_max = np.max(losses_per_probe[:, len(losses_per_probe[0]) // 2:])
        plt.ylim(0, combined_max * 2)  # Cap the y-axis dynamically for combined plot
        
        # Save the combined plot
        plt.savefig(os.path.join(save_path, 'training_test_losses_plot.png'), dpi=300, bbox_inches='tight')
        plt.clf()  # Clear the figure


    
    
    
    
    #verify my reconstruction of the model is correct    
    # timesteps = 100000
    # correct = 0
    # for i in range(timesteps):
    #     if i!=0 and (i % (timesteps // 20) == 0 or i == timesteps - 1):
    #         print(100 * i / timesteps, "%", "Runtime:", time.time()-start, "Projected Runtime:", (timesteps-i)/(i/(time.time() - start)))
        
    #     action_dist, _ = reconstruct_model(obs)
    #     model_pred = int(model.predict(obs, deterministic=True)[0])
    #     reconstructed_pred = int(torch.argmax(action_dist)) 
        
    #     if model_pred != reconstructed_pred:
    #         print(f"OH GOD! THE MODEL PREDICTED {model_pred}, BUT OUR RECONSTRUCTED MODEL PREDICTED {reconstructed_pred}! PANIC PANIC PANIC!")
    #     else:
    #         correct += 1
    #     obs, reward, done, info = env.step(model_pred)
    #     if done:
    #       obs = env.reset()
          
    # print("Total accuracy of the reconstructed model:", correct/timesteps)
    # 100% accuracy with timesteps = 100000. There were some errors with 1000000, but that was before I counted them, so... eh. This is good enough to continue
    