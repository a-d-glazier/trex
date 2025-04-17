from envs import *
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.types import Trajectory
from torch.utils import data 
from imitation.algorithms.adversarial.gail import GAIL
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from gym_simplegrid.envs.simple_grid import SimpleGridEnv
from envs.customgrid import CustomSimpleGrid
from imitation.rewards.reward_nets import BasicRewardNet
from services.models.simple import GailNet
from services.rl.value_iteration import value_iteration
from services.rl.policy import get_stochastic_policy, get_deterministic_policy

class EarlyStopping:
    def __init__(self, patience: int, delta: float = 0):
        """
        :param patience: How many epochs with no improvement after which training will be stopped.
        :param delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def check_early_stop(self, current_score: float, epoch: int):
        """
        Check if early stopping should be triggered.
        :param current_score: Current score (e.g., validation loss).
        :param epoch: Current epoch number.
        :return: True if training should stop, False otherwise.
        """
        if self.best_score is None or current_score < self.best_score - self.delta:
            self.best_score = current_score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            return True
        return False


# Example usage of early stopping
early_stopping = EarlyStopping(patience=5, delta=0.01)  # Stop if no improvement in 5 steps
# 1. Define the Expert Policy (Optimal Policy for SimpleGridEnv)
# class ExpertPolicy:
#     def __init__(self, goal_xy):
#         self.goal_xy = goal_xy  # Assume goal position is known
    
#     def act(self, obs):
#         row, col = obs // 3, obs % 3
#         goal_row, goal_col = self.goal_xy

#         if row < goal_row: return 1  # Move Down
#         if row > goal_row: return 0  # Move Up
#         if col < goal_col: return 3  # Move Right
#         if col > goal_col: return 2  # Move Left
#         return 0  # Default action

class ExpertPolicy:
    def __init__(self, goal_xy):
        self.goal_xy = goal_xy  # Assume goal position is known

    def act(self, obs):
        """
        Takes a one-hot encoded state as input and returns a one-hot encoded action.
        """
        # Convert one-hot state to discrete index
        state_index = np.argmax(obs)
        row, col = state_index // 3, state_index % 3
        goal_row, goal_col = self.goal_xy

        # Determine the action (one-hot encoding)
        # action = np.zeros(4, dtype=int)
        if state_index == 0:
            # random down or right
            action = np.random.choice([1, 3])
        elif state_index == 1:
            # random down or right
            action=np.random.choice([1, 3])
        elif state_index == 2:
            # down
            action=1
        elif state_index == 3:
            # random down or right
            action=np.random.choice([1, 3])
        elif state_index == 4:
            # random down or right
            action=np.random.choice([1, 3])
        elif state_index == 5:
            # down
            action=1
        elif state_index == 6:
            # right
            action=3
        elif state_index == 7:
            # right
            action=3

        # if state_index == 0:
        #     # random down or right
        #     action[np.random.choice([1, 3])] = 1
        # elif state_index == 1:
        #     # random down or right
        #     action[np.random.choice([1, 3])] = 1
        # elif state_index == 2:
        #     # down
        #     action[1] = 1
        # elif state_index == 3:
        #     # random down or right
        #     action[np.random.choice([1, 3])] = 1
        # elif state_index == 4:
        #     # random down or right
        #     action[np.random.choice([1, 3])] = 1
        # elif state_index == 5:
        #     # down
        #     action[1] = 1
        # elif state_index == 6:
        #     # right
        #     action[3] = 1
        # elif state_index == 7:
        #     # right
        #     action[3] = 1

        # if row < goal_row:
        #     action[1] = 1  # Move Down
        # elif row > goal_row:
        #     action[0] = 1  # Move Up
        # elif col < goal_col:
        #     action[3] = 1  # Move Right
        # elif col > goal_col:
        #     action[2] = 1  # Move Left
        # else:
        #     action[0] = 1  # Default action (Up)

        return action

# 2. Generate Expert Demonstrations
def collect_expert_trajectories(env, expert, num_episodes=100):
    trajectories = []
    for _ in range(num_episodes):
        obs, _ = env.reset(options={'start_loc':0, 'goal_loc':8})
        acts, rews, infos, dones = [], [], [], []
        observations = [obs]

        terminal = False
        while True:
            action = expert.act(obs)
            obs, reward, done, _, info = env.step(action)
            acts.append(action)
            rews.append(reward)
            infos.append(info)
            dones.append(done)
            observations.append(obs)
            if done:
                terminal = True
                break

        trajectories.append(Trajectory(
            obs=np.array(observations),
            acts=np.array(acts),
            infos=np.array(infos),
            terminal=terminal
        ))

    return trajectories

# 3. Setup Environment & Expert
env = gym.make("DummyGrid-v0")
expert_policy = ExpertPolicy(goal_xy=(2,2))
dataset = collect_expert_trajectories(env, expert_policy, num_episodes=300)

# 4. Train GAIL Agent
vec_env = DummyVecEnv([lambda: gym.make("DummyGrid-v0")])
learner = PPO("MlpPolicy", 
              vec_env, 
              verbose=1, 
              batch_size=128,         # Batch size
              learning_rate=0.001,   # Learning rate (starting low)
              gamma=0.99,             # Discount factor
              n_epochs=5,             # Number of epochs per update
              clip_range=0.2,         # Clip range for PPO
              vf_coef=0.5,            # Value function coefficient
              ent_coef=0.01,          # Entropy coefficient
              policy_kwargs={"net_arch": [32, 32]},  # Policy network architecture
            #   n_steps=5,
              seed=7)
# PPO("MlpPolicy", vec_env, verbose=1, batch_size=64, learning_rate=0.005, gamma=0.99, seed=7)
# reward_net = BasicRewardNet(env.unwrapped.observation_space, env.unwrapped.action_space, use_next_state=True)
reward_net = GailNet(env.unwrapped.observation_space, env.unwrapped.action_space)

gail_trainer = GAIL(
    venv=vec_env,
    demonstrations=dataset,
    gen_algo=learner,
    n_disc_updates_per_round=2,
    demo_batch_size=128,  # Set batch size
    gen_replay_buffer_capacity=512,
    reward_net=reward_net,
    allow_variable_horizon=True
    # gen_train_timesteps=2048,
)

# Set up early stopping
early_stopping = EarlyStopping(patience=5, delta=0.01)

gail_trainer.train(30000)  # Train for k steps

# for epoch in range(100000):
#     gail_trainer.train(2048)  
#     current_loss = gail_trainer.get_descriminator_loss()
#     # Check early stopping condition
#     if early_stopping.check_early_stop(current_loss, epoch):
#         print(f"Early stopping triggered at epoch {epoch}.")
#         break

# 5. Save & Evaluate
learner.save("gail_agent")
env = gym.make(
    'DummyGrid-v0', 
    render_mode='human'
)

def get_reward_matrix(reward, num_states, num_actions):
    R = np.zeros((num_states, num_actions, num_states))
    for s in range(num_states):
        s_onehot = np.zeros((1,num_states))
        s_onehot[0][s] = 1
        for a in range(num_actions):
            a_onehot = np.zeros((1,num_actions))
            a_onehot[0][a] = 1
            for s_prime in range(num_states):
                s_prime_onehot = np.zeros((1,num_states))
                s_prime_onehot[0][s_prime] = 1
                R[s, a, s_prime] = reward_net(torch.tensor(s_onehot, dtype=torch.float32), torch.tensor(a_onehot, dtype=torch.float32), torch.tensor(s_prime_onehot, dtype=torch.float32)).item()
    return R

R = get_reward_matrix(reward_net, env.unwrapped.n_states, len(env.unwrapped.MOVES))
v, q , _ = value_iteration(
    env.unwrapped.get_transition_probabilities(), 
    R,
    discount=0.9
)
policy, policy_exec = get_deterministic_policy(q)
nonpolicy, nonpolicy_exec = get_stochastic_policy(q)

observations, info = env.reset(options={'start_loc':0, 'goal_loc':8})
for _ in range(100):
    action = policy_exec(observations.argmax())
    observations, reward, done, _, info = env.step(action)
    if done:
        observations, info = env.reset(options={'start_loc':0, 'goal_loc':8})


# observations, info = env.reset(options={'start_loc':0, 'goal_loc':8})
# for _ in range(100):
#     # action = learner.policy(torch.tensor([observations]))[0][0].item()
#     action = learner.predict(observations, deterministic=True)[0].item()
#     observations, reward, done, _, info = env.step(action)
#     if done:
#         observations, info = env.reset()
#         # break