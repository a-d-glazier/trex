"""
Maximum Margin Inverse Reinforcement Learning implementation.

This module implements the maximum margin IRL algorithm, which learns a
reward function that maximizes the margin between expert and non-expert
demonstrations while maintaining a consistent policy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union, Any
import random

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

class MaxMarginIRL:
    """
    Implementation of Maximum Margin Inverse Reinforcement Learning (MaxMargin IRL)
    using direct policy optimization with correct projection method from the original algorithm.
    """
    
    def __init__(
        self,
        env_name: str = 'CartPole-v1',
        gamma: float = 0.99,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Create environment
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.gamma = gamma
        self.device = device
        
        # Get state and action dimensions from the environment
        self.state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n 
            self.discrete_actions = True
            self.n_actions = self.env.action_space.n
        else:
            raise ValueError("Only discrete action spaces are supported")
        
        # Initialize feature dimension
        self.n_features = 17  # Number of features in get_state_features
        
        # Generate expert demonstrations
        self.expert_trajectories = self._generate_expert_demonstrations()
        
        # Compute state normalization parameters from expert trajectories
        all_states = []
        for traj in self.expert_trajectories:
            all_states.extend(traj["states"])
        all_states = np.array(all_states)
        self.state_mean = np.mean(all_states, axis=0)
        self.state_std = np.std(all_states, axis=0) + 1e-8
        
        # Initialize policy network and optimizer
        self.policy = PolicyNetwork(self.state_dim, self.n_actions).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
        # Initialize reward weights with smaller values
        self.reward_weights = torch.randn(self.n_features, device=device) * 0.01
        self.reward_weights.requires_grad_(True)
        self.reward_optimizer = optim.Adam([self.reward_weights], lr=5e-5)
        
        # Initialize exploration parameters
        self.exploration_rate = 0.3  # Start with higher exploration
        self.min_exploration_rate = 0.01
        
        # Compute expert feature expectations once
        print("Computing expert feature expectations...")
        self.expert_feature_expectations = self.compute_feature_expectations(self.expert_trajectories)
        
        # Initialize feature expectations bar with random policy
        print("Initializing feature expectations...")
        sampled_trajectories = self.sample_trajectories(len(self.expert_trajectories))
        self.feature_expectations_bar = self.compute_feature_expectations(sampled_trajectories)
        
    def _generate_expert_demonstrations(self, n_demos: int = 100) -> List[Dict[str, np.ndarray]]:
        """Generate expert demonstrations using a PID controller."""
        expert_demos = []
        
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
        
        for _ in range(n_demos):
            state, _ = self.env.reset()
            states = [state]
            actions = []
            done = False
            
            while not done:
                action = expert_policy(state)
                next_state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                actions.append(action)
                state = next_state
                states.append(state)
                
                if len(states) >= 500:  # Maximum episode length
                    break
            
            expert_demos.append({
                "states": np.array(states[:-1]),  # Exclude final state
                "actions": np.array(actions),
                "next_states": np.array(states[1:])  # Exclude initial state
            })
        
        return expert_demos
        
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
        Optimize the policy network using the current reward function with improved stability.
        """
        total_loss = 0.0
        n_updates = 0
        batch_size = 64  # Increased batch size
        
        # Collect all state-action pairs and rewards
        all_states = []
        all_actions = []
        all_rewards = []
        all_returns = []  # Track returns for advantage computation
        
        for traj in trajectories:
            states = traj["states"]
            actions = traj["actions"]
            env_rewards = traj["rewards"]  # Use environment rewards for better learning
            
            # Compute returns
            returns = []
            R = 0
            for r in reversed(env_rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            
            # Add to training data
            for t, (state, action, R) in enumerate(zip(states, actions, returns)):
                all_states.append(state)
                all_actions.append(action)
                all_returns.append(R)
                
                # Also compute IRL reward
                reward = self.compute_reward(state)
                all_rewards.append(reward)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(all_states)).to(self.device)
        actions_tensor = torch.LongTensor(np.array(all_actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.array(all_rewards)).to(self.device)
        returns_tensor = torch.FloatTensor(np.array(all_returns)).to(self.device)
        
        # Normalize rewards and returns
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Combine IRL rewards and environment returns
        combined_rewards = 0.5 * rewards_tensor + 0.5 * returns_tensor
        
        # Multiple epochs of training
        n_samples = len(all_states)
        indices = list(range(n_samples))
        
        for epoch in range(5):
            random.shuffle(indices)
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                state_batch = states_tensor[batch_indices]
                action_batch = actions_tensor[batch_indices]
                reward_batch = combined_rewards[batch_indices]
                
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
    
    def train(self, n_iterations: int = 1000, n_trajectories_per_iter: int = 100, reward_threshold: float = 450, eval_episodes: int = 10) -> Dict:
        """
        Main training loop for MaxMargin IRL with improved stability.
        """
        stats = {
            'eval_rewards': [],
            'reward_losses': [],
            'policy_losses': []
        }
        best_reward = 0
        patience = 20
        no_improvement_count = 0
        
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
            
            # Early stopping with patience
            if avg_reward > best_reward:
                best_reward = avg_reward
                no_improvement_count = 0
                if best_reward >= reward_threshold:
                    print(f"Reached reward threshold of {reward_threshold}. Early stopping.")
                    break
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= patience:
                print(f"No improvement for {patience} iterations. Early stopping.")
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