"""
Malik's Inverse Constraint Reinforcement Learning (ICRL) implementation.

This module implements the ICRL algorithm, which learns a reward function
from expert demonstrations using constraint-based optimization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union, Any
import random

class RewardNetwork(nn.Module):
    """Neural network to represent the reward function."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.scaled_sigmoid = lambda x: 0.1 + 0.8 * torch.sigmoid(x)  # Outputs between 0.1 and 0.9
    
    def forward(self, states, actions):
        """Forward pass through the reward network."""
        x = torch.cat([states, actions], dim=1)
        return self.scaled_sigmoid(self.network(x))

class Malik:
    """
    Implementation of Inverse Constraint Reinforcement Learning (ICRL) algorithm
    compatible with Gymnasium environments.
    """
    
    def __init__(
        self,
        env_name: str = 'CartPole-v1',
        gamma: float = 0.99,
        backward_iterations: int = 2,
        forward_kl_threshold: float = 0.01,
        reverse_kl_threshold: float = 0.01,
        reward_lr: float = 0.0001,
        batch_size: int = 64,
        hidden_dim: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Create environment
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.gamma = gamma
        self.backward_iterations = backward_iterations
        self.forward_kl_threshold = forward_kl_threshold
        self.reverse_kl_threshold = reverse_kl_threshold
        self.batch_size = batch_size
        self.device = device
        
        # Get state and action dimensions from the environment
        self.state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n 
            self.discrete_actions = True
            self.n_actions = self.env.action_space.n
        else:
            self.action_dim = self.env.action_space.shape[0]
            self.discrete_actions = False
        
        # Initialize reward networks
        self.reward_network = RewardNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.prev_reward_network = RewardNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        
        # Initialize prev_reward_network with perturbed weights
        with torch.no_grad():
            for param in self.prev_reward_network.parameters():
                param.data += torch.randn_like(param) * 0.01
        
        # Setup optimizer
        self.reward_optimizer = optim.Adam(self.reward_network.parameters(), lr=reward_lr)
        
        # Generate expert demonstrations
        self.expert_trajectories = self._generate_expert_demonstrations()
        
        # For policy learning
        self.policy = None
        
        # Calculate state normalization parameters from expert demonstrations
        all_states = np.vstack([traj["states"] for traj in self.expert_trajectories])
        self.state_mean = np.mean(all_states, axis=0)
        self.state_std = np.std(all_states, axis=0) + 1e-8
        
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

    def learn_policy(self) -> Dict[str, Any]:
        """
        Learn a policy by solving equation (9) in the paper:
        min_λ≥0 max_φ J(π^φ) + (1/β)H(π^φ) - λ(E_{τ~π^φ}[ζ_θ(τ)] - α)
        """
        
        class Policy(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim, discrete, device="cpu"):
                super().__init__()
                self.discrete = discrete
                self.device = device
                
                self.network = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
                
                if discrete:
                    self.action_head = nn.Linear(hidden_dim, action_dim)
                    self.action_head.weight.data.mul_(0.1)
                    self.action_head.bias.data.mul_(0.1)
                else:
                    self.mean_head = nn.Linear(hidden_dim, action_dim)
                    self.log_std_head = nn.Linear(hidden_dim, action_dim)
                    self.log_std_min = -20
                    self.log_std_max = 2
            
            def forward(self, state):
                x = self.network(state)
                if self.discrete:
                    logits = self.action_head(x)
                    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
                    logits_stable = logits - logits_max
                    action_probs = torch.softmax(logits_stable, dim=-1)
                    action_probs = torch.nan_to_num(action_probs, nan=1.0/logits.shape[-1])
                    action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                    return action_probs
                else:
                    mean = self.mean_head(x)
                    log_std = torch.clamp(self.log_std_head(x), self.log_std_min, self.log_std_max)
                    return mean, torch.exp(log_std)

            def sample_action(self, state):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                if self.discrete:
                    probs = self.forward(state)
                    probs = torch.nan_to_num(probs, nan=1.0/probs.shape[-1])
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    try:
                        action = torch.multinomial(probs, 1).item()
                    except RuntimeError:
                        action = random.randint(0, probs.shape[-1] - 1)
                    return action
                else:
                    mean, std = self.forward(state)
                    normal = torch.distributions.Normal(mean, std)
                    action = normal.sample()
                    return action.detach().cpu().numpy().squeeze()

            def sample_greedy_action(self, policy, state):
                """Sample greedy action from policy."""
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    if self.discrete:
                        action_probs = self.forward(state_tensor)
                        action = torch.argmax(action_probs, dim=1).item()
                    else:
                        mean, _ = self.forward(state_tensor)
                        action = mean.cpu().numpy().squeeze()
                
                return action
        
        # Initialize policy network
        policy = Policy(self.state_dim, self.action_dim if not self.discrete_actions else self.n_actions, 
                       128, self.discrete_actions, self.device).to(self.device)
        policy_optimizer = optim.Adam(policy.parameters(), lr=0.001)

        # Initialize Lagrange multiplier λ
        lambda_param = torch.tensor(0.1, device=self.device, requires_grad=True)
        lambda_optimizer = optim.Adam([lambda_param], lr=0.01)

        # Training parameters
        entropy_coef = 0
        alpha = 0.01
        max_episodes = 1000
        max_steps = 100
        
        for episode in range(max_episodes):
            state, _ = self.env.reset()
            episode_states = []
            episode_actions = []
            episode_log_probs = []
            episode_rewards = []
            episode_entropies = []
            
            for step in range(max_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    if self.discrete_actions:
                        probs = policy(state_tensor)
                        m = torch.distributions.Categorical(probs)
                        action = m.sample().item()
                        log_prob = m.log_prob(torch.tensor(action, device=self.device))
                        entropy = m.entropy()
                    else:
                        mean, std = policy(state_tensor)
                        m = torch.distributions.Normal(mean, std)
                        action = m.sample().squeeze()
                        log_prob = m.log_prob(action).sum(dim=-1)
                        entropy = m.entropy().sum(dim=-1)
                        action = action.cpu().numpy()
                
                next_state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_states.append(state)
                episode_log_probs.append(log_prob.item())
                episode_entropies.append(entropy.item())
                
                if self.discrete_actions:
                    action_tensor = torch.zeros(self.n_actions)
                    action_tensor[action] = 1
                    episode_actions.append(action_tensor.numpy())
                else:
                    episode_actions.append(action)
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                if self.discrete_actions:
                    action_tensor = torch.zeros(1, self.n_actions).to(self.device)
                    action_tensor[0, action] = 1
                else:
                    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    reward = self.reward_network(state_tensor, action_tensor).item()
                
                episode_rewards.append(reward)
                state = next_state
                
                if done:
                    break
            
            states = torch.FloatTensor(np.array(episode_states)).to(self.device)
            if self.discrete_actions:
                actions = torch.FloatTensor(np.array(episode_actions)).to(self.device)
                action_probs = policy(states)
                dist = torch.distributions.Categorical(action_probs)
                log_probs = dist.log_prob(actions.max(1)[1])
                entropy = dist.entropy()
            else:
                actions = torch.FloatTensor(np.array(episode_actions)).to(self.device)
                mean, std = policy(states)
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(actions).sum(dim=1)
                entropy = dist.entropy().sum(dim=1)

            rewards = torch.FloatTensor(episode_rewards).to(self.device)

            returns = []
            G = 0
            for r in reversed(episode_rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns).to(self.device)

            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            expected_reward = rewards.mean()
            entropy_term = entropy.mean()
            expected_cost = (1 - rewards).mean()
            
            policy_loss = -(log_probs * returns).mean() - entropy_coef * entropy_term + lambda_param * (expected_cost - alpha)

            policy_optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            policy_loss.backward()
            policy_optimizer.step()

            lambda_loss = -lambda_param * (expected_cost - alpha)

            lambda_optimizer.zero_grad()
            lambda_loss.backward()
            lambda_optimizer.step()

            with torch.no_grad():
                lambda_param.data.clamp_(min=0.0)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{max_episodes}, "
                      f"Avg Reward: {expected_reward.item():.3f}, "
                      f"Entropy: {entropy_term.item():.3f}, "
                      f"Lambda: {lambda_param.item():.3f}")

            if episode > 10 and expected_cost.item() <= alpha:
                print(f"Policy satisfied reward constraint at episode {episode+1}")
                break
        
        self.policy = policy
        return {"policy": policy}

    def sample_trajectories(self, policy: nn.Module, n_trajectories: int) -> List[Dict[str, np.ndarray]]:
        """Sample trajectories from the current policy."""
        trajectories = []
        
        for _ in range(n_trajectories):
            state, _ = self.env.reset()
            states = []
            actions = []
            next_states = []
            done = False
            
            while not done:
                states.append(state)
                action = policy.sample_action(state)
                
                next_state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                if self.discrete_actions:
                    action_array = np.zeros(self.n_actions)
                    action_array[action] = 1
                    actions.append(action_array)
                else:
                    actions.append(action)
                
                next_states.append(next_state)
                state = next_state
                
                if len(states) >= 1000:
                    break
            
            trajectory = {
                "states": np.array(states),
                "actions": np.array(actions),
                "next_states": np.array(next_states)
            }
            trajectories.append(trajectory)
        
        return trajectories

    def compute_importance_weights(self, sampled_trajectory: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute importance sampling weights using equation (10)."""
        states = torch.FloatTensor(sampled_trajectory["states"]).to(self.device)
        actions = torch.FloatTensor(sampled_trajectory["actions"]).to(self.device)
        
        with torch.no_grad():
            zeta_theta = self.reward_network(states, actions).squeeze()
            zeta_theta_bar = self.prev_reward_network(states, actions).squeeze()
            
            zeta_theta = torch.clamp(zeta_theta, 1e-6, 1.0 - 1e-6)
            zeta_theta_bar = torch.clamp(zeta_theta_bar, 1e-6, 1.0 - 1e-6)

        log_weights = []
        for zeta, zeta_bar in zip(zeta_theta, zeta_theta_bar):
            log_weight = np.log(zeta.item()) - np.log(zeta_bar.item())
            log_weight = np.clip(log_weight, -10.0, 10.0)
            log_weights.append(log_weight)
        
        log_trajectory_weight = np.sum(log_weights)
        trajectory_weight = np.clip(np.exp(log_trajectory_weight), 1e-6, 1e6)
        
        return trajectory_weight

    def update_reward_parameters(self, sampled_trajectories: List[Dict[str, np.ndarray]], 
                             weights: np.ndarray) -> None:
        """Update reward parameters using gradient descent based on equation (11)."""
        N = len(self.expert_trajectories)
        expert_states_list = []
        expert_actions_list = []
        
        for traj in self.expert_trajectories:
            states = traj["states"]
            actions = traj["actions"]
            
            for s, a in zip(states, actions):
                expert_states_list.append(s)
                
                if self.discrete_actions:
                    if np.isscalar(a) or (isinstance(a, np.ndarray) and a.size == 1):
                        a_idx = int(a) if np.isscalar(a) else int(a.item())
                        a_onehot = np.zeros(self.n_actions)
                        a_onehot[a_idx] = 1
                        expert_actions_list.append(a_onehot)
                    else:
                        expert_actions_list.append(a)
                else:
                    expert_actions_list.append(np.array(a).flatten())
        
        M = len(sampled_trajectories)
        sampled_states_list = []
        sampled_actions_list = []
        sampled_weights_list = []
        
        for i, traj in enumerate(sampled_trajectories):
            traj_weight = weights[i]
            states = traj["states"]
            actions = traj["actions"]
            
            for s, a in zip(states, actions):
                sampled_states_list.append(s)
                sampled_actions_list.append(a)
                sampled_weights_list.append(traj_weight)
        
        if expert_states_list:
            expert_states = torch.FloatTensor(np.array(expert_states_list)).to(self.device)
            expert_actions = torch.FloatTensor(np.array(expert_actions_list)).to(self.device)
        else:
            expert_states = torch.FloatTensor([]).to(self.device)
            expert_actions = torch.FloatTensor([]).to(self.device)
        
        if sampled_states_list:
            sampled_states = torch.FloatTensor(np.array(sampled_states_list)).to(self.device)
            sampled_actions = torch.FloatTensor(np.array(sampled_actions_list)).to(self.device)
            sampled_weights = torch.FloatTensor(np.array(sampled_weights_list)).to(self.device)
        else:
            sampled_states = torch.FloatTensor([]).to(self.device)
            sampled_actions = torch.FloatTensor([]).to(self.device)
            sampled_weights = torch.FloatTensor([]).to(self.device)
        
        expert_batch_size = min(self.batch_size, len(expert_states)) if len(expert_states) > 0 else 0
        sampled_batch_size = min(self.batch_size, len(sampled_states)) if len(sampled_states) > 0 else 0
        
        n_expert_batches = (len(expert_states) + expert_batch_size - 1) // expert_batch_size if expert_batch_size > 0 else 0
        n_sampled_batches = (len(sampled_states) + sampled_batch_size - 1) // sampled_batch_size if sampled_batch_size > 0 else 0
        
        n_epochs = 5
        
        for epoch in range(n_epochs):
            if len(expert_states) > 0:
                expert_indices = torch.randperm(len(expert_states))
            if len(sampled_states) > 0:
                sampled_indices = torch.randperm(len(sampled_states))
            
            for batch_idx in range(max(n_expert_batches, n_sampled_batches)):
                self.reward_optimizer.zero_grad()
                total_loss = 0.0
                
                if batch_idx < n_expert_batches and len(expert_states) > 0:
                    start_idx = batch_idx * expert_batch_size
                    end_idx = min(start_idx + expert_batch_size, len(expert_states))
                    
                    batch_expert_states = expert_states[expert_indices[start_idx:end_idx]]
                    batch_expert_actions = expert_actions[expert_indices[start_idx:end_idx]]
                    
                    expert_rewards = self.reward_network(batch_expert_states, batch_expert_actions)
                    expert_log_zeta = torch.log(expert_rewards + 1e-8)
                    expert_loss = -torch.sum(expert_log_zeta) / (N + 1e-8)
                    total_loss += expert_loss
                
                if batch_idx < n_sampled_batches and len(sampled_states) > 0:
                    start_idx = batch_idx * sampled_batch_size
                    end_idx = min(start_idx + sampled_batch_size, len(sampled_states))
                    
                    batch_sampled_states = sampled_states[sampled_indices[start_idx:end_idx]]
                    batch_sampled_actions = sampled_actions[sampled_indices[start_idx:end_idx]]
                    batch_weights = sampled_weights[sampled_indices[start_idx:end_idx]]
                    
                    sampled_rewards = self.reward_network(batch_sampled_states, batch_sampled_actions)
                    sampled_log_zeta = torch.log(sampled_rewards + 1e-8)
                    sampled_loss = torch.sum(batch_weights.unsqueeze(1) * sampled_log_zeta) / (M + 1e-8)
                    total_loss += sampled_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.reward_network.parameters(), max_norm=1.0)
                self.reward_optimizer.step()
                
        l2_reg = 0.01
        for param in self.reward_network.parameters():
            total_loss += l2_reg * torch.sum(param ** 2)

    def compute_kl_divergences(self, sampled_trajectories: List[Dict[str, np.ndarray]],
                           weights: np.ndarray) -> Tuple[float, float]:
        """Compute forward and reverse KL divergences using equation (12)."""
        mean_weight = np.mean(weights) if len(weights) > 0 else 1e-8
        forward_kl = 2 * np.log(mean_weight + 1e-8)
        
        log_weights = np.log(weights + 1e-8)
        weighted_log_weights = (weights - mean_weight) * log_weights
        reverse_kl = np.sum(weighted_log_weights) / (mean_weight + 1e-8)
        
        return forward_kl, reverse_kl

    def train(self, n_iterations: int = 1000, n_trajectories_per_iter: int = 100, reward_threshold: float = 450, eval_episodes: int = 10) -> Dict:
        """Main training loop for ICRL algorithm."""
        stats = {
            'forward_kls': [],
            'reverse_kls': [],
            'eval_rewards': []
        }
        early_stop = False
        
        print("Starting ICRL training...")
        
        for i in range(n_iterations):
            print(f"Outer iteration {i+1}/{n_iterations}")
            
            print("Learning policy...")
            policy_result = self.learn_policy()
            current_policy = policy_result["policy"]
            
            eval_rewards = []
            for _ in range(eval_episodes):
                state, _ = self.env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action = current_policy.sample_greedy_action(current_policy, state)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    state = next_state
                
                eval_rewards.append(episode_reward)
            
            avg_reward = np.mean(eval_rewards)
            stats['eval_rewards'].append(avg_reward)
            
            print(f"  Evaluation: Avg reward over {eval_episodes} episodes: {avg_reward:.2f}")
        
            if avg_reward >= reward_threshold:
                print(f"  Reached reward threshold of {reward_threshold}. Early stopping.")
                early_stop = True
                break
            
            for j in range(self.backward_iterations):
                print(f"  Backward iteration {j+1}/{self.backward_iterations}")
                
                self.prev_reward_network.load_state_dict(self.reward_network.state_dict())
                before_update_norm = sum(p.norm().item() for p in self.reward_network.parameters())
                
                print("  Sampling trajectories...")
                sampled_trajectories = self.sample_trajectories(current_policy, n_trajectories_per_iter)
                
                print("  Computing importance weights...")
                weights = np.array([self.compute_importance_weights(traj) for traj in sampled_trajectories])
                
                if not np.all(np.isfinite(weights)):
                    non_finite_count = np.sum(~np.isfinite(weights))
                    print(f"Warning: {non_finite_count} non-finite weight values detected")
                    weights = np.array([w if np.isfinite(w) else 1.0 for w in weights])
                
                print(f"  Weight stats - min: {np.min(weights):.6e}, max: {np.max(weights):.6e}, mean: {np.mean(weights):.6e}")
                
                print("  Updating reward parameters...")
                self.update_reward_parameters(sampled_trajectories, weights)
                
                after_update_norm = sum(p.norm().item() for p in self.reward_network.parameters())
                print(f"  Parameter norm before update: {before_update_norm:.6f}, after update: {after_update_norm:.6f}")
                
                forward_kl, reverse_kl = self.compute_kl_divergences(sampled_trajectories, weights)
                stats['forward_kls'].append(forward_kl)
                stats['reverse_kls'].append(reverse_kl)
                
                print(f"  Forward KL: {forward_kl:.6f}, Reverse KL: {reverse_kl:.6f}")
                
                if forward_kl >= self.forward_kl_threshold or reverse_kl >= self.reverse_kl_threshold:
                    print(f"  Converged at backward iteration {j+1}!")
                    break

                if forward_kl == 0 and reverse_kl == 0:
                    print(f"  Converged at backward iteration {j+1}!")
                    break
        
        if not early_stop:
            final_policy_result = self.learn_policy()
        else:
            final_policy_result = policy_result
        self.policy = final_policy_result["policy"]
        
        return {
            "policy": self.policy,
            "reward_network": self.reward_network,
            "stats": stats
        }
    
    def save(self, path: str) -> None:
        """Save the trained models."""
        torch.save({
            "reward_network": self.reward_network.state_dict(),
            "policy": self.policy.state_dict() if self.policy else None
        }, path)
    
    def load(self, path: str) -> None:
        """Load trained models."""
        checkpoint = torch.load(path)
        self.reward_network.load_state_dict(checkpoint["reward_network"])
        if checkpoint["policy"] and self.policy:
            self.policy.load_state_dict(checkpoint["policy"])

    def normalize_state(self, state):
        """Normalize the state using expert demonstration statistics."""
        if isinstance(state, np.ndarray):
            return (state - self.state_mean) / self.state_std
        elif isinstance(state, torch.Tensor):
            state_mean = torch.FloatTensor(self.state_mean).to(state.device)
            state_std = torch.FloatTensor(self.state_std).to(state.device)
            return (state - state_mean) / state_std 