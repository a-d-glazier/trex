"""
Maximum Entropy Inverse Reinforcement Learning implementation.

This module implements the maximum entropy IRL algorithm, which learns a
reward function that maximizes the likelihood of expert demonstrations
while maintaining maximum entropy over the policy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union, Any, Callable
import random

class PolicyNetwork(nn.Module):
    """Simple policy network for CartPole."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        return torch.softmax(logits, dim=-1)

class MaxEntIRL:
    """
    Implementation of Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL)
    compatible with Gymnasium environments.
    """
    
    def __init__(
        self,
        env_name: str = 'CartPole-v1',
        gamma: float = 0.99,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        expert_policy: Optional[Callable] = None,
        seed: int = 42  # Add seed parameter with default value
    ):
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        
        # Create environment and set its seed
        self.env = gym.make(env_name)
        self.env.reset(seed=seed)  # Set environment seed
        self.env_name = env_name
        self.gamma = gamma
        self.device = device
        self.expert_policy = expert_policy
        
        # Get state and action dimensions from the environment
        self.state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n 
            self.discrete_actions = True
            self.n_actions = self.env.action_space.n
        else:
            raise ValueError("Only discrete action spaces are supported")
        
        # Initialize reward parameters
        self.reward_params = np.ones(self.state_dim)  # One parameter per state dimension
        
        # Set whether to normalize features based on environment
        self.normalize_features = 'grid' not in env_name.lower() # TODO: not the best way to do it
        
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
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        
        print(f"Feature normalization: {'enabled' if self.normalize_features else 'disabled'}")
        
    def _generate_expert_demonstrations(self, n_demos: int = 100) -> List[Dict[str, np.ndarray]]:
        """Generate expert demonstrations using the provided expert policy."""
        print(f"\nGenerating {n_demos} expert demonstrations...")
        expert_demos = []
        total_steps = 0
        total_actions = defaultdict(int)
        
        for i in range(n_demos):
            state, _ = self.env.reset()
            states = [state]
            actions = []
            done = False
            
            while not done:
                # Use the provided expert policy
                if self.expert_policy is not None:
                    action = self.expert_policy(state)
                else:
                    # Fallback to random actions if no expert policy provided
                    action = self.env.action_space.sample()
                
                next_state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                actions.append(action)
                total_actions[action] += 1
                state = next_state
                states.append(state)
                
                if len(states) >= 500:  # Maximum episode length
                    break
            
            total_steps += len(actions)
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{n_demos} demonstrations, avg steps: {total_steps/(i+1):.1f}")
            
            expert_demos.append({
                "states": np.array(states[:-1]),  # Exclude final state
                "actions": np.array(actions),
                "next_states": np.array(states[1:])  # Exclude initial state
            })
        
        print("\nExpert demonstration statistics:")
        print(f"Average trajectory length: {total_steps/n_demos:.1f} steps")
        print("Action distribution:")
        action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        total_actions_count = sum(total_actions.values())
        for action, count in total_actions.items():
            print(f"  {action_names[action]}: {count/total_actions_count*100:.1f}%")
        print()
        
        return expert_demos
        
    def normalize_state(self, state):
        """Normalize state using precomputed statistics."""
        return (state - self.state_mean) / self.state_std
        
    def compute_reward(self, state: np.ndarray) -> float:
        """Compute reward for a state using feature-based reward function."""
        normalized_state = self.normalize_state(state)
        return np.dot(normalized_state, self.reward_params)
    
    def update_reward_parameters(self, expert_trajectories: List[Dict[str, np.ndarray]], 
                               sampled_trajectories: List[Dict[str, np.ndarray]]) -> float:
        """
        Update reward parameters using gradient descent based on MaxEnt IRL loss.
        
        Args:
            expert_trajectories: List of expert demonstration trajectories
            sampled_trajectories: List of trajectories sampled from current policy
            
        Returns:
            float: The gradient norm (loss)
        """
        # Compute feature expectations from expert trajectories
        expert_features = []
        for traj in expert_trajectories:
            if self.normalize_features:
                states = np.array([self.normalize_state(s) for s in traj["states"]])
            else:
                states = np.array(traj["states"])
            expert_features.extend(states)
        expert_features = np.array(expert_features)
        expert_feature_expectation = np.mean(expert_features, axis=0)
        
        # Compute feature expectations from sampled trajectories
        sampled_features = []
        for traj in sampled_trajectories:
            if self.normalize_features:
                states = np.array([self.normalize_state(s) for s in traj["states"]])
            else:
                states = np.array(traj["states"])
            sampled_features.extend(states)
        sampled_features = np.array(sampled_features)
        sampled_feature_expectation = np.mean(sampled_features, axis=0)
        
        # Compute gradient
        grad = expert_feature_expectation - sampled_feature_expectation
        
        # Update reward parameters
        lr = 0.1
        self.reward_params += lr * grad
        
        # Normalize reward parameters
        self.reward_params = self.reward_params / np.linalg.norm(self.reward_params)
        
        return np.linalg.norm(grad)  # Return gradient norm as loss
    
    def update_policy(self, trajectories: List[Dict[str, np.ndarray]]) -> float:
        """Update policy using policy gradient with entropy regularization."""
        states = []
        actions = []
        rewards = []
        
        # Collect states, actions, and compute rewards using learned reward function
        for traj in trajectories:
            states.extend(traj["states"])
            actions.extend(traj["actions"])
            # Use learned reward function instead of environment reward
            rewards.extend([self.compute_reward(s) for s in traj["states"]])
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        # Normalize rewards for better training stability
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Get action probabilities
        probs = self.policy(states)
        
        # Compute policy loss with entropy regularization
        log_probs = torch.log(probs[torch.arange(len(actions)), actions])
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # Policy gradient loss with entropy regularization
        policy_loss = -torch.mean(log_probs * rewards + 0.01 * entropy)
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)  # Gradient clipping
        self.policy_optimizer.step()
        
        return policy_loss.item()
    
    def sample_trajectories(self, n_trajectories: int) -> List[Dict[str, np.ndarray]]:
        """Sample trajectories using the current policy."""
        print(f"\nSampling {n_trajectories} trajectories using current policy...")
        trajectories = []
        total_steps = 0
        total_actions = defaultdict(int)
        
        for i in range(n_trajectories):
            state, _ = self.env.reset()
            states = []
            actions = []
            next_states = []
            done = False
            
            while not done:
                states.append(state)
                
                # Get action from policy network with multinomial sampling for exploration
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(self.normalize_state(state)).to(self.device)
                    probs = self.policy(state_tensor.unsqueeze(0)).squeeze(0)
                    # Use multinomial sampling during training for better exploration
                    action = torch.multinomial(probs, 1).item()
                
                # Execute action
                next_state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                actions.append(action)
                total_actions[action] += 1
                next_states.append(next_state)
                state = next_state
                
                if len(states) >= 500:  # Maximum trajectory length
                    break
            
            total_steps += len(actions)
            if (i + 1) % 20 == 0:
                print(f"Sampled {i + 1}/{n_trajectories} trajectories, avg steps: {total_steps/(i+1):.1f}")
            
            trajectory = {
                "states": np.array(states),
                "actions": np.array(actions),
                "next_states": np.array(next_states)
            }
            trajectories.append(trajectory)
        
        print("\nSampled trajectory statistics:")
        print(f"Average trajectory length: {total_steps/n_trajectories:.1f} steps")
        print("Action distribution:")
        action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        total_actions_count = sum(total_actions.values())
        for action, count in total_actions.items():
            print(f"  {action_names[action]}: {count/total_actions_count*100:.1f}%")
        print()
        
        return trajectories
    
    def train(self, n_iterations: int = 1000, n_trajectories_per_iter: int = 100, reward_threshold: float = 350, eval_episodes: int = 10) -> Dict:
        """
        Main training loop for MaxEnt IRL.
        
        Args:
            n_iterations: Number of training iterations
            n_trajectories_per_iter: Number of trajectories to sample per iteration
            reward_threshold: Early stopping threshold for average reward
            eval_episodes: Number of evaluation episodes per iteration
            
        Returns:
            Dictionary with training statistics and learned models
        """
        stats = {
            'eval_rewards': [],
            'reward_losses': [],
            'policy_losses': []
        }
        early_stop = False
        
        print("\n" + "="*50)
        print("Starting MaxEnt IRL training...")
        print(f"Environment: {self.env_name}")
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {self.n_actions}")
        print(f"Device: {self.device}")
        print("="*50 + "\n")
        
        for i in range(n_iterations):
            print(f"\nIteration {i+1}/{n_iterations}")
            print("-"*30)
            
            # Sample trajectories from current policy
            sampled_trajectories = self.sample_trajectories(n_trajectories_per_iter)
            
            # Update reward parameters
            print("\nUpdating reward parameters...")
            reward_loss = self.update_reward_parameters(self.expert_trajectories, sampled_trajectories)
            print(f"Reward loss: {reward_loss:.4f}")
            
            # Update policy
            print("\nUpdating policy...")
            policy_loss = self.update_policy(sampled_trajectories)
            print(f"Policy loss: {policy_loss:.4f}")
            
            # Store losses
            stats['reward_losses'].append(reward_loss)
            stats['policy_losses'].append(policy_loss)
            
            # Evaluate current policy
            print("\nEvaluating current policy...")
            eval_rewards = []
            eval_lengths = []
            
            for ep in range(eval_episodes):
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(self.normalize_state(state)).to(self.device)
                        probs = self.policy(state_tensor.unsqueeze(0)).squeeze(0)
                        action = torch.argmax(probs).item()
                    
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    episode_length += 1
                    state = next_state
                
                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)
            
            avg_reward = np.mean(eval_rewards)
            avg_length = np.mean(eval_lengths)
            stats['eval_rewards'].append(avg_reward)
            
            print(f"\nEvaluation results over {eval_episodes} episodes:")
            print(f"  Average reward: {avg_reward:.2f} (±{np.std(eval_rewards):.2f})")
            print(f"  Average length: {avg_length:.1f} (±{np.std(eval_lengths):.1f})")
            print(f"  Min/Max reward: {min(eval_rewards):.1f}/{max(eval_rewards):.1f}")
            
            # Early stopping check based on reward
            if avg_reward >= reward_threshold:
                print(f"\nReached reward threshold of {reward_threshold}. Early stopping.")
                early_stop = True
                break
        
        if not early_stop:
            print("\nTraining completed without reaching reward threshold.")
        
        print("\nFinal training statistics:")
        print(f"  Best average reward: {max(stats['eval_rewards']):.2f}")
        print(f"  Final reward loss: {stats['reward_losses'][-1]:.4f}")
        print(f"  Final policy loss: {stats['policy_losses'][-1]:.4f}")
        
        return {
            "policy": self.policy,
            "reward_params": self.reward_params,
            "stats": stats
        } 