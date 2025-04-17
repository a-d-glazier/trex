import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union, Any
import random
from envs import *

def value_iteration(p_transition, reward, discount=0.99, eps=1e-4):
    """
    Value iteration algorithm for policy learning.
    
    Args:
        p_transition: Transition probabilities [state, action, next_state]
        reward: Reward function [state, action]
        discount: Discount factor
        eps: Convergence threshold
        
    Returns:
        Value function and Q-function
    """
    n_states, n_actions = reward.shape
    v = np.zeros(n_states)
    q = np.zeros((n_states, n_actions))
    
    while True:
        v_old = v.copy()
        
        # Update Q-values
        for s in range(n_states):
            for a in range(n_actions):
                q[s, a] = reward[s, a] + discount * np.sum(p_transition[s, a] * v)
        
        # Update value function
        v = np.max(q, axis=1)
        
        # Check convergence
        if np.max(np.abs(v - v_old)) < eps:
            break
    
    return v, q

def get_stochastic_policy(q, temperature=1.0):
    """
    Convert Q-values to a stochastic policy.
    
    Args:
        q: Q-function [state, action]
        temperature: Temperature parameter for softmax
        
    Returns:
        Policy function and policy execution function
    """
    def policy(state):
        # Apply softmax with temperature and numerical stability
        q_state = q[state]
        q_max = np.max(q_state)
        exp_q = np.exp((q_state - q_max) / temperature)
        probs = exp_q / np.sum(exp_q)
        return probs
    
    def policy_exec(state):
        # Sample action from policy
        probs = policy(state)
        return int(np.random.choice(len(probs), p=probs))  # Return integer action
    
    return policy, policy_exec

def discretize_state(state, n_states=1000):
    """
    Discretize a continuous state into a discrete state index.
    For CartPole, we focus on the angle and angular velocity.
    """
    x, x_dot, theta, theta_dot = state
    
    # Normalize theta to [-1, 1]
    theta_norm = max(min(theta / (np.pi/4), 1.0), -1.0)
    
    # Normalize theta_dot to [-1, 1]
    theta_dot_norm = max(min(theta_dot / 4.0, 1.0), -1.0)
    
    # Normalize x to [-1, 1]
    x_norm = max(min(x / 2.4, 1.0), -1.0)
    
    # Normalize x_dot to [-1, 1]
    x_dot_norm = max(min(x_dot / 4.0, 1.0), -1.0)
    
    # Create a more informative state representation
    # We'll divide each dimension into 5 bins, giving us 625 total states
    n_bins = 5
    x_bin = int((x_norm + 1) * (n_bins - 1) / 2)
    x_dot_bin = int((x_dot_norm + 1) * (n_bins - 1) / 2)
    theta_bin = int((theta_norm + 1) * (n_bins - 1) / 2)
    theta_dot_bin = int((theta_dot_norm + 1) * (n_bins - 1) / 2)
    
    # Combine bins into a single index
    state_idx = (x_bin * n_bins**3 + x_dot_bin * n_bins**2 + 
                theta_bin * n_bins + theta_dot_bin)
    
    return state_idx

def get_state_features(state):
    """
    Get features for a state that capture important aspects of the CartPole dynamics.
    """
    x, x_dot, theta, theta_dot = state
    
    # Enhanced features for better performance
    features = [
        1.0,  # Bias term
        x,  # Cart position
        x_dot,  # Cart velocity
        theta,  # Pole angle
        theta_dot,  # Pole angular velocity
        x * x,  # Quadratic features
        theta * theta,
        x_dot * x_dot,
        theta_dot * theta_dot,
        x * theta,  # Cross terms
        x_dot * theta_dot,
        np.sin(theta),  # Trigonometric features
        np.cos(theta),
        np.tanh(x),  # Nonlinear features
        np.tanh(theta),
        np.tanh(x_dot),
        np.tanh(theta_dot)
    ]
    
    return np.array(features)

class PolicyNetwork(nn.Module):
    """Enhanced policy network for CartPole."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        return torch.softmax(logits, dim=-1)

def feature_expectation_from_trajectories(feature_matrix, trajectories):
    """
    Compute feature expectations from trajectories.
    
    Args:
        feature_matrix: Matrix of state features [n_states, n_features]
        trajectories: List of trajectories, each containing states and actions
        
    Returns:
        Feature expectations vector
    """
    feature_expectations = np.zeros(feature_matrix.shape[1])
    n_trajectories = len(trajectories)
    
    for traj in trajectories:
        states = traj["states"]
        for state in states:
            state_idx = discretize_state(state)
            feature_expectations += feature_matrix[state_idx]
    
    return feature_expectations / n_trajectories

def get_reward_matrix(reward, num_states, num_actions):
    """Convert state rewards to state-action rewards."""
    R = np.zeros((num_states, num_actions))
    for s in range(num_states):
        for a in range(num_actions):
            R[s, a] = reward[s]
    return R

class MaxMarginIRL:
    """
    Implementation of Maximum Margin Inverse Reinforcement Learning (MaxMargin IRL)
    using direct policy optimization with correct projection method from the original algorithm.
    """
    
    def __init__(
        self,
        env: gym.Env,
        expert_trajectories: List[Dict[str, np.ndarray]],
        gamma: float = 0.99,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.env = env
        self.expert_trajectories = expert_trajectories
        self.gamma = gamma
        self.device = device
        
        # Get state and action dimensions from the environment
        self.state_dim = env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = env.action_space.n 
            self.discrete_actions = True
            self.n_actions = env.action_space.n
        else:
            raise ValueError("Only discrete action spaces are supported")
        
        # Initialize feature dimension
        self.n_features = 17  # Number of features in get_state_features
        
        # Initialize policy network and optimizer
        self.policy = PolicyNetwork(self.state_dim, self.n_actions).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
        # Initialize state normalization parameters
        all_states = []
        for traj in expert_trajectories:
            all_states.extend(traj["states"])
        all_states = np.array(all_states)
        self.state_mean = np.mean(all_states, axis=0)
        self.state_std = np.std(all_states, axis=0) + 1e-8
        
        # Initialize reward weights with smaller values
        self.reward_weights = torch.randn(self.n_features, device=device) * 0.01
        self.reward_weights.requires_grad_(True)
        self.reward_optimizer = optim.Adam([self.reward_weights], lr=5e-5)
        
        # Initialize exploration parameters
        self.exploration_rate = 0.3  # Start with higher exploration
        self.min_exploration_rate = 0.01
        
        # Compute expert feature expectations once
        print("Computing expert feature expectations...")
        self.expert_feature_expectations = self.compute_feature_expectations(expert_trajectories)
        
        # Initialize feature expectations bar with random policy
        print("Initializing feature expectations...")
        sampled_trajectories = self.sample_trajectories(len(expert_trajectories))
        self.feature_expectations_bar = self.compute_feature_expectations(sampled_trajectories)
        
    def normalize_state(self, state):
        """Normalize state using precomputed statistics."""
        return (state - self.state_mean) / self.state_std
        
    def compute_reward(self, state: np.ndarray) -> float:
        """Compute reward for a state using feature-based reward function with normalization."""
        features = get_state_features(state)
        features_tensor = torch.FloatTensor(features).to(self.device)
        # Normalize reward to be between -1 and 1
        reward = torch.tanh(torch.dot(features_tensor, self.reward_weights))
        return reward.item()
    
    def compute_feature_expectations(self, trajectories: List[Dict[str, np.ndarray]]) -> torch.Tensor:
        """
        Compute feature expectations from trajectories using discounted sum.
        """
        feature_expectations = torch.zeros(self.n_features, device=self.device)
        n_trajectories = len(trajectories)
        
        for traj in trajectories:
            states = traj["states"]
            discounted_sum = torch.zeros(self.n_features, device=self.device)
            
            for t, state in enumerate(states):
                features = get_state_features(state)
                features_tensor = torch.FloatTensor(features).to(self.device)
                discounted_sum += (self.gamma ** t) * features_tensor
            
            feature_expectations += discounted_sum
        
        return feature_expectations / n_trajectories
    
    def update_reward_parameters(self, expert_trajectories: List[Dict[str, np.ndarray]], 
                               sampled_trajectories: List[Dict[str, np.ndarray]], 
                               epsilon: float = 0.1) -> float:
        """
        Update reward parameters using MaxMargin IRL algorithm with projection method.
        """
        # Compute feature expectations for current policy
        current_feature_expectations = self.compute_feature_expectations(sampled_trajectories)
        
        # Compute the difference between expert and current policy
        diff = self.expert_feature_expectations - current_feature_expectations
        
        # Update feature expectations bar using projection method
        updated_loss = current_feature_expectations - self.feature_expectations_bar
        projection_term = (updated_loss * torch.dot(updated_loss, self.reward_weights)) / (torch.norm(updated_loss) ** 2 + 1e-8)
        self.feature_expectations_bar += projection_term
        
        # Update reward weights
        self.reward_weights = self.expert_feature_expectations - self.feature_expectations_bar
        
        # Compute margin
        margin = torch.norm(self.reward_weights)
        
        return margin.item()
    
    def sample_action(self, state_tensor, deterministic=False):
        """Sample action from policy with epsilon-greedy exploration."""
        with torch.no_grad():
            probs = self.policy(state_tensor)
            if deterministic or random.random() > self.exploration_rate:
                return torch.argmax(probs).item()
            else:
                return random.randint(0, self.n_actions - 1)
    
    def sample_trajectories(self, n_trajectories: int, deterministic=False) -> List[Dict[str, np.ndarray]]:
        """Sample trajectories using the current policy with epsilon-greedy exploration."""
        trajectories = []
        
        for _ in range(n_trajectories):
            state, _ = self.env.reset()
            states = []
            actions = []
            next_states = []
            rewards = []  # Track environment rewards for normalization
            done = False
            
            while not done:
                states.append(state)
                
                # Get normalized state
                state_tensor = torch.FloatTensor(self.normalize_state(state)).to(self.device).unsqueeze(0)
                
                # Sample action
                action = self.sample_action(state_tensor, deterministic)
                
                # Execute action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                actions.append(action)
                next_states.append(next_state)
                rewards.append(reward)
                state = next_state
                
                if len(states) >= 500:  # Maximum trajectory length
                    break
            
            trajectory = {
                "states": np.array(states),
                "actions": np.array(actions),
                "next_states": np.array(next_states),
                "rewards": np.array(rewards)
            }
            trajectories.append(trajectory)
        
        return trajectories
    
    def optimize_policy(self, trajectories: List[Dict[str, np.ndarray]]) -> float:
        """
        Optimize the policy network using only the learned reward function.
        """
        total_loss = 0.0
        n_updates = 0
        batch_size = 64  # Increased batch size

        # Collect all state-action pairs and rewards
        all_states = []
        all_actions = []
        all_rewards = []

        for traj in trajectories:
            states = traj["states"]
            actions = traj["actions"]

            # Add to training data
            for state, action in zip(states, actions):
                all_states.append(state)
                all_actions.append(action)

                # Compute IRL reward
                reward = self.compute_reward(state)
                all_rewards.append(reward)

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(all_states)).to(self.device)
        actions_tensor = torch.LongTensor(np.array(all_actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.array(all_rewards)).to(self.device)

        # Normalize rewards
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Multiple epochs of training
        n_samples = len(all_states)
        indices = list(range(n_samples))

        for epoch in range(5):
            random.shuffle(indices)
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]

                state_batch = states_tensor[batch_indices]
                action_batch = actions_tensor[batch_indices]
                reward_batch = rewards_tensor[batch_indices] # Use only IRL rewards

                # Get normalized states
                state_batch_norm = torch.FloatTensor(self.normalize_state(state_batch.cpu().numpy())).to(self.device)

                # Get action probabilities
                probs = self.policy(state_batch_norm)

                # Compute policy loss with entropy regularization
                log_probs = torch.log(probs[range(len(batch_indices)), action_batch] + 1e-8)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                loss = -torch.mean(log_probs * reward_batch + 0.01 * entropy)

                # Update policy
                self.policy_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        return total_loss / n_updates if n_updates > 0 else 0.0
    
    def train(self, n_iterations: int, n_trajectories_per_iter: int, reward_threshold: float = 450, eval_episodes: int = 10) -> Dict:
        """
        Main training loop for MaxMargin IRL without early stopping based on lack of improvement.
        """
        stats = {
            'eval_rewards': [],
            'reward_losses': [],
            'policy_losses': []
        }
        best_reward = 0
        
        print("Starting MaxMargin IRL training...")
        
        for i in range(n_iterations):
            print(f"Iteration {i+1}/{n_iterations}")
            
            # Sample more trajectories
            n_trajectories = n_trajectories_per_iter * (i // 10 + 1)
            sampled_trajectories = self.sample_trajectories(n_trajectories)
            
            # Update reward parameters
            print("Updating reward parameters...")
            reward_loss = self.update_reward_parameters(
                self.expert_trajectories,
                sampled_trajectories,
                epsilon=0.1
            )
            
            # Optimize policy multiple times with different batches
            print("Optimizing policy...")
            policy_loss = 0
            for _ in range(10):
                policy_loss += self.optimize_policy(sampled_trajectories)
            policy_loss /= 10
            
            # Store losses
            stats['reward_losses'].append(reward_loss)
            stats['policy_losses'].append(policy_loss)
            
            # Evaluate current policy
            eval_rewards = []
            for _ in range(eval_episodes):
                state, _ = self.env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    state_tensor = torch.FloatTensor(self.normalize_state(state)).to(self.device).unsqueeze(0)
                    action = self.sample_action(state_tensor, deterministic=True)
                    
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    state = next_state
                
                eval_rewards.append(episode_reward)
            
            avg_reward = np.mean(eval_rewards)
            stats['eval_rewards'].append(avg_reward)
            
            print(f"Iteration {i+1}: Avg reward over {eval_episodes} episodes: {avg_reward:.2f}")
            print(f"Reward Loss: {reward_loss:.4f}, Policy Loss: {policy_loss:.4f}")
            
            # Early stopping based only on reward threshold
            if avg_reward > best_reward:
                best_reward = avg_reward
                if best_reward >= reward_threshold:
                    print(f"Reached reward threshold of {reward_threshold}. Early stopping.")
                    break
            
            # Anneal exploration rate
            if i > 0 and i % 10 == 0:
                self.exploration_rate = max(
                    self.min_exploration_rate,
                    self.exploration_rate * 0.95
                )
        
        return {
            "policy": self.policy,
            "reward_weights": self.reward_weights,
            "stats": stats,
            "best_reward": best_reward
        }

def process_expert_demonstrations(demos: List[Dict]) -> List[Dict[str, np.ndarray]]:
    """
    Process expert demonstrations into the format expected by MaxMargin IRL.
    """
    processed_trajectories = []
    
    for demo in demos:
        if "observations" in demo and "actions" in demo:
            states = np.array(demo["observations"][:-1])
            actions = np.array(demo["actions"])
            next_states = np.array(demo["observations"][1:])
            
            min_len = min(len(states), len(actions), len(next_states))
            
            processed_trajectory = {
                "states": states[:min_len],
                "actions": actions[:min_len],
                "next_states": next_states[:min_len]
            }
            processed_trajectories.append(processed_trajectory)
    
    return processed_trajectories

def example_usage():
    """Example of how to use the MaxMargin IRL implementation with Gymnasium."""
    print(f"Current PyTorch seed: {torch.initial_seed()}")
    print(f"Current NumPy RNG state: {np.random.get_state()[1][0]}")
    
    # Create a Gymnasium environment
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    
    # Generate expert demonstrations
    def expert_policy(state):
        """Expert policy using PID-like control for CartPole."""
        x, x_dot, theta, theta_dot = state
        
        # Constants for the controller
        Kp_theta = 10.0  # Proportional gain for angle
        Kd_theta = 2.0   # Derivative gain for angle
        Kp_x = 0.5       # Proportional gain for position
        Kd_x = 1.0       # Derivative gain for velocity
        
        # Compute the angle error term (we want theta = 0)
        angle_error = theta
        angle_rate_error = theta_dot
        
        # Compute the position error term (we want x = 0)
        position_error = x
        velocity_error = x_dot
        
        # Combine terms to get control signal
        u = Kp_theta * angle_error + Kd_theta * angle_rate_error + \
            Kp_x * position_error + Kd_x * velocity_error
        
        # Convert continuous control to discrete action
        return 1 if u > 0 else 0
    
    # Generate expert demonstrations
    n_expert_demos = 100
    expert_demos = []
    
    for _ in range(n_expert_demos):
        state, _ = env.reset()
        states = [state]
        actions = []
        done = False
        
        while not done:
            action = expert_policy(state)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            actions.append(action)
            state = next_state
            states.append(state)
            
            if len(states) >= 500:  # Maximum episode length
                break
        
        expert_demos.append({
            "observations": states,
            "actions": actions
        })
    
    # Process expert demonstrations
    expert_trajectories = process_expert_demonstrations(expert_demos)
    
    # Evaluate expert demonstrations
    print("\nEvaluating expert demonstrations...")
    expert_rewards = []
    for _ in range(10):  # Evaluate 10 episodes
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = expert_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
        
        expert_rewards.append(episode_reward)
    
    avg_expert_reward = np.mean(expert_rewards)
    print(f"Expert demonstration average reward: {avg_expert_reward:.2f}")
    
    if avg_expert_reward < 250:
        print("Warning: Expert demonstrations do not meet target performance (250).")
        print("Please improve the expert policy before proceeding with training.")
        return None
    
    print("Expert demonstrations meet target performance. Proceeding with training...\n")
    
    # Create and train MaxMargin IRL
    maxmargin_irl = MaxMarginIRL(
        env=env,
        expert_trajectories=expert_trajectories,
        gamma=0.99
    )
    
    # Train MaxMargin IRL with early stopping
    results = maxmargin_irl.train(
        n_iterations=1000, 
        n_trajectories_per_iter=100,
        reward_threshold=350,
        eval_episodes=10
    )
    
    # Evaluate the learned policy
    learned_policy = results["policy"]
    eval_rewards = []
    
    for _ in range(10):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(maxmargin_irl.normalize_state(state)).to(maxmargin_irl.device)
                probs = learned_policy(state_tensor.unsqueeze(0)).squeeze(0)
                action = torch.argmax(probs).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
        
        eval_rewards.append(episode_reward)
    
    print(f"Average evaluation reward: {np.mean(eval_rewards)}")
    
    # Render a few episodes with the learned policy
    env = gym.make(env_name, render_mode="human")
    
    for episode in range(3):  # Render 3 episodes
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(maxmargin_irl.normalize_state(state)).to(maxmargin_irl.device)
                probs = learned_policy(state_tensor.unsqueeze(0)).squeeze(0)
                action = torch.argmax(probs).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            env.render()  # Render the environment
        
        print(f"Episode {episode + 1} reward: {episode_reward}")
    
    env.close()
    
    return results

if __name__ == "__main__":
    example_usage() 