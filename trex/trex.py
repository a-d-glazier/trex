import gymnasium as gym

import numpy as np
import scipy.special
import torch

from envs import *
from services.rl.value_iteration import value_iteration
from services.trajectory import generate_demonstrations, rank_trajectories_by_reward
from services.train import create_training_data, learn_reward
from services.models.simple import Net
from services.predict import predict_traj_return
from services.rl.policy import get_stochastic_policy

np.random.seed(7)

env = gym.make(
    'DummyGrid-v0', 
    render_mode='human'
)
observations, info = env.reset(options={'start_loc':0, 'goal_loc':8})

# Calculating value functions
v, q, training_trajectories = value_iteration(
    env.unwrapped.get_transition_probabilities(), 
    env.unwrapped.get_reward_function(), 
    discount=0.9,
    checkpoint_trajectories=True,
    env=env
)
policy, policy_exec = get_stochastic_policy(q)

flattened_trajectories = [item for obj in training_trajectories for item in obj.trajectories]
ranked_trajectories = rank_trajectories_by_reward(flattened_trajectories, env)

X, y = create_training_data(ranked_trajectories)

lr = 0.00005
weight_decay = 0.0
num_iter = 50 #num times through training data
l1_reg=0.00
reward_model_path = './reward_model.pth'
# Now we create a reward network and optimize it using the training data.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
reward_net = Net()
reward_net.to(device)
optimizer = torch.optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
learn_reward(reward_net, optimizer, X, y, num_iter, l1_reg, reward_model_path)
#save reward network
torch.save(reward_net.state_dict(), reward_model_path)
print()

with torch.no_grad():
    for traj in ranked_trajectories:
        predicted_return = predict_traj_return(reward_net, traj[0].transitions())
        actual_return = sum([env.unwrapped.get_reward(*env.unwrapped.to_xy(next_s)) for s, a, next_s in traj[0].transitions()])
        print(f"Predicted: {predicted_return}, Actual: {actual_return}, Trajectory: {traj}")


r = lambda s: reward_net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
# Function to generate the reward matrix
def get_reward_matrix(r, num_states, num_actions):
    R = np.zeros((num_states, num_actions, num_states))  # Initialize matrix with zeros
    
    for s in range(num_states):
        for a in range(num_actions):
            for s_prime in range(num_states):
                R[s, a, s_prime] = r((s, a, s_prime))  # Populate the matrix with rewards from the lambda function
    
    return R

R = get_reward_matrix(r, 9, 4)
min_val = np.min(R)
max_val = np.max(R)
normalized_R = (R - min_val) / (max_val - min_val)
scaled_R = 2 * normalized_R - 1
R = scaled_R
v, q , _ = value_iteration(
    env.unwrapped.get_transition_probabilities(), 
    R,
    discount=0.9
)

policy, policy_exec = get_stochastic_policy(q)

deterministic_policy = np.zeros_like(policy)
max_indices = np.argmax(policy, axis=1)
deterministic_policy[np.arange(policy.shape[0]), max_indices] = 1
deterministic_policy_exec = lambda state: np.random.choice([*range(deterministic_policy.shape[1])], p=deterministic_policy[state, :])

env.reset(options={'start_loc':0, 'goal_loc':8})
for _ in range(100):
    # action = env.action_space.sample()
    # action = policy_exec(observations)
    action = deterministic_policy_exec(observations)
    observations, reward, done, _, info = env.step(action)
    if done:
        observations, info = env.reset(options={'start_loc':0, 'goal_loc':8})
        # break