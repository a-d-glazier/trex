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
    
    # Combine the two most important features into a single index
    state_val = (theta_norm + 1) / 2 + (theta_dot_norm + 1) / 4
    state_idx = min(int(state_val * (n_states - 1)), n_states - 1)
    return max(0, state_idx)  # Ensure index is within bounds

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
        # Add numerical stability to softmax
        logits = self.net(x)
        logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]  # Subtract max for numerical stability
        return torch.softmax(logits, dim=-1)

class MaxEntIRL:
    """
    Implementation of Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL)
    compatible with Gymnasium environments.
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
        
        # Initialize reward parameters
        self.reward_params = np.ones(self.state_dim)  # One parameter per state dimension
        
        # Compute state normalization parameters from expert trajectories
        all_states = []
        for traj in expert_trajectories:
            all_states.extend(traj["states"])
        all_states = np.array(all_states)
        self.state_mean = np.mean(all_states, axis=0)
        self.state_std = np.std(all_states, axis=0) + 1e-8
        
        # Initialize policy network and optimizer
        self.policy = PolicyNetwork(self.state_dim, self.n_actions).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        
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
        """
        # Compute feature expectations from expert trajectories
        expert_features = []
        for traj in expert_trajectories:
            states = np.array([self.normalize_state(s) for s in traj["states"]])
            expert_features.extend(states)
        expert_features = np.array(expert_features)
        expert_feature_expectation = np.mean(expert_features, axis=0)
        
        # Compute feature expectations from sampled trajectories
        sampled_features = []
        for traj in sampled_trajectories:
            states = np.array([self.normalize_state(s) for s in traj["states"]])
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
        trajectories = []
        
        for _ in range(n_trajectories):
            state, _ = self.env.reset()
            states = []
            actions = []
            next_states = []
            done = False
            
            while not done:
                states.append(state)
                
                # Get action from policy network with numerical stability
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(self.normalize_state(state)).to(self.device)
                    logits = self.policy.net(state_tensor.unsqueeze(0)).squeeze(0)
                    # Add numerical stability
                    logits = logits - torch.max(logits)
                    probs = torch.softmax(logits, dim=-1)
                    # Ensure valid probabilities
                    probs = torch.clamp(probs, min=1e-6)
                    probs = probs / probs.sum()
                    action = torch.multinomial(probs, 1).item()
                
                # Execute action
                next_state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                actions.append(action)
                next_states.append(next_state)
                state = next_state
                
                if len(states) >= 500:  # Maximum trajectory length
                    break
            
            trajectory = {
                "states": np.array(states),
                "actions": np.array(actions),
                "next_states": np.array(next_states)
            }
            trajectories.append(trajectory)
        
        return trajectories
    
    def train(self, n_iterations: int, n_trajectories_per_iter: int, reward_threshold: float = 250, eval_episodes: int = 10) -> Dict:
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
        
        print("Starting MaxEnt IRL training...")
        
        for i in range(n_iterations):
            print(f"Iteration {i+1}/{n_iterations}")
            
            # Sample trajectories from current policy
            print("Sampling trajectories...")
            sampled_trajectories = self.sample_trajectories(n_trajectories_per_iter)
            
            # Update reward parameters
            print("Updating reward parameters...")
            reward_loss = self.update_reward_parameters(self.expert_trajectories, sampled_trajectories)
            
            # Update policy
            print("Updating policy...")
            policy_loss = self.update_policy(sampled_trajectories)
            
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
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(self.normalize_state(state)).to(self.device)
                        probs = self.policy(state_tensor.unsqueeze(0)).squeeze(0)
                        # Use greedy action selection for evaluation
                        action = torch.argmax(probs).item()
                    
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    state = next_state
                
                eval_rewards.append(episode_reward)
            
            avg_reward = np.mean(eval_rewards)
            stats['eval_rewards'].append(avg_reward)
            
            print(f"Iteration {i+1}: Avg reward over {eval_episodes} episodes: {avg_reward:.2f}")
            print(f"Reward Loss: {reward_loss:.4f}, Policy Loss: {policy_loss:.4f}")
            
            # Early stopping check based on reward
            if avg_reward >= reward_threshold:
                print(f"Reached reward threshold of {reward_threshold}. Early stopping.")
                early_stop = True
                break
        
        return {
            "policy": self.policy,
            "reward_params": self.reward_params,
            "stats": stats
        }

def process_expert_demonstrations(demos: List[Dict]) -> List[Dict[str, np.ndarray]]:
    """
    Process expert demonstrations into the format expected by MaxEnt IRL.
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
    """Example of how to use the MaxEnt IRL implementation with Gymnasium."""
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
    
    # Create and train MaxEnt IRL
    maxent_irl = MaxEntIRL(
        env=env,
        expert_trajectories=expert_trajectories,
        gamma=0.99
    )
    
    # Train MaxEnt IRL with early stopping
    results = maxent_irl.train(
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
                state_tensor = torch.FloatTensor(maxent_irl.normalize_state(state)).to(maxent_irl.device)
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
                state_tensor = torch.FloatTensor(maxent_irl.normalize_state(state)).to(maxent_irl.device)
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