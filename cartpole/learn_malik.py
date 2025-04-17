import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union, Any
import random
from envs import *


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
        # Add sigmoid to constrain outputs between 0 and 1
        # self.sigmoid = nn.Sigmoid()
        # Instead of using standard sigmoid that can saturate at 0,
        # use a scaled sigmoid that has a minimum value of 0.1
        self.scaled_sigmoid = lambda x: 0.1 + 0.8 * torch.sigmoid(x)  # Outputs between 0.1 and 0.9
    
    def forward(self, states, actions):
        """Forward pass through the reward network."""
        x = torch.cat([states, actions], dim=1)
        return self.scaled_sigmoid(self.network(x))


class ICRL:
    """
    Implementation of Inverse Constraint Reinforcement Learning (ICRL) algorithm
    compatible with Gymnasium environments.
    """
    
    def __init__(
        self,
        env: gym.Env,
        expert_trajectories: List[Dict[str, np.ndarray]],
        backward_iterations: int = 10,
        forward_kl_threshold: float = 1,
        reverse_kl_threshold: float = 1,
        reward_lr: float = 0.0001,
        batch_size: int = 64,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.env = env
        self.expert_trajectories = expert_trajectories
        self.backward_iterations = backward_iterations
        self.forward_kl_threshold = forward_kl_threshold
        self.reverse_kl_threshold = reverse_kl_threshold
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        
        # Get state and action dimensions from the environment
        self.state_dim = env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = env.action_space.n 
            self.discrete_actions = True
            self.n_actions = env.action_space.n
        else:
            self.action_dim = env.action_space.shape[0]
            self.discrete_actions = False
        
        # Initialize reward network for current parameters θ
        self.reward_network = RewardNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
         # Initialize reward network for previous parameters θ̄ with perterbation so they aren't exactly the same
        self.prev_reward_network = RewardNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        with torch.no_grad():
            for param in self.prev_reward_network.parameters():
                param.data += torch.randn_like(param) * 0.01 
        
        # Setup optimizers
        self.reward_optimizer = optim.Adam(self.reward_network.parameters(), lr=reward_lr)
        
        # For policy learning
        self.policy = None
    
    def preprocess_trajectory(self, trajectory: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert numpy arrays to PyTorch tensors and move to device."""
        states = torch.FloatTensor(trajectory["states"]).to(self.device)
        
        if self.discrete_actions:
            # One-hot encode discrete actions
            actions_raw = trajectory["actions"].reshape(-1)
            actions = torch.zeros(len(actions_raw), self.n_actions)
            actions[range(len(actions_raw)), actions_raw.astype(int)] = 1
        else:
            actions = torch.FloatTensor(trajectory["actions"]).to(self.device)
        
        next_states = torch.FloatTensor(trajectory["next_states"]).to(self.device)
        return states, actions, next_states
    
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
                    # Initialize the action head with smaller weights to prevent extreme values
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
                    # Apply a softmax with better numerical stability
                    # Subtract max for numerical stability before softmax
                    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
                    logits_stable = logits - logits_max
                    action_probs = torch.softmax(logits_stable, dim=-1)
                    # Ensure no NaN values and probs sum to 1
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
                    # Ensure we have valid probabilities
                    probs = torch.nan_to_num(probs, nan=1.0/probs.shape[-1])
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    try:
                        action = torch.multinomial(probs, 1).item()
                    except RuntimeError:
                        # Fallback if there are still issues
                        print("Warning: Using fallback action sampling")
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
                        # Take action with highest probability
                        action = torch.argmax(action_probs, dim=1).item()
                    else:
                        mean, _ = self.forward(state_tensor)  # For continuous actions, ignore std dev
                        action = mean.cpu().numpy().squeeze()
                
                return action
                
        # Initialize policy network
        policy = Policy(self.state_dim, self.action_dim if not self.discrete_actions else self.n_actions, 
                        128, self.discrete_actions).to(self.device)
        policy_optimizer = optim.Adam(policy.parameters(), lr=0.001)

        # Initialize Lagrange multiplier λ
        lambda_param = torch.tensor(0.1, device=self.device, requires_grad=True)
        lambda_optimizer = optim.Adam([lambda_param], lr=0.01)

        # Entropy regularization coefficient (1/β)
        entropy_coef = 0 #0.01
        # Constraint threshold α (target expected reward)
        alpha = 0.01
        
        # Training loop for policy learning
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
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # Sample action from policy
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
                
                # Execute action in environment
                next_state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                episode_states.append(state)
                episode_log_probs.append(log_prob.item())
                episode_entropies.append(entropy.item())
                
                if self.discrete_actions:
                    action_tensor = torch.zeros(self.n_actions)
                    action_tensor[action] = 1
                    episode_actions.append(action_tensor.numpy())
                else:
                    episode_actions.append(action)
                
                # Compute reward using reward network
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                if self.discrete_actions:
                    action_tensor = torch.zeros(1, self.n_actions).to(self.device)
                    action_tensor[0, action] = 1
                else:
                    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # ζ_θ(s_t, a_t) - reward given by the reward network
                    reward = self.reward_network(state_tensor, action_tensor).item()
                
                episode_rewards.append(reward)
                state = next_state
                
                if done:
                    break
            
            # Convert to tensors for policy update
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

            # Compute rewards 
            rewards = torch.FloatTensor(episode_rewards).to(self.device)

            # Compute returns (discounted rewards)
            returns = []
            G = 0
            for r in reversed(episode_rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns).to(self.device)

            # Normalize returns for stability
            if len(returns) > 1:  # Only normalize if we have more than one return
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Compute expected reward (E_{τ~π^φ}[ζ_θ(τ)])
            expected_reward = rewards.mean()

            # Compute entropy term
            entropy_term = entropy.mean()

            # Compute policy loss from equation (9)
            # J(π^φ) + (1/β)H(π^φ) - λ(E_{τ~π^φ}[ζ_θ(τ)] - α)
            # For maximizing rewards, we use negative log_probs * returns

            # Compute expected cost (E_{τ~π^φ}[ζ̄_θ(τ)])
            expected_cost = (1 - rewards).mean()  # 1 - reward is the cost
            policy_loss = -(log_probs * returns).mean() - entropy_coef * entropy_term + lambda_param * (expected_cost - alpha)

            # Update policy (max_φ part of equation 9)
            policy_optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            policy_loss.backward()
            policy_optimizer.step()

            # Update λ parameter (min_λ≥0 part of equation 9)
            lambda_loss = -lambda_param * (expected_cost - alpha)

            lambda_optimizer.zero_grad()
            lambda_loss.backward()
            lambda_optimizer.step()

            # Project λ to be non-negative
            with torch.no_grad():
                lambda_param.data.clamp_(min=0.0)

            # Print progress
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{max_episodes}, "
                    f"Avg Reward: {expected_reward.item():.3f}, "
                    f"Entropy: {entropy_term.item():.3f}, "
                    f"Lambda: {lambda_param.item():.3f}")

            # Check convergence
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
                
                # Get action from policy
                action = policy.sample_action(state)
                
                # Execute action
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
                
                if len(states) >= 1000:  # Maximum trajectory length
                    break
            
            trajectory = {
                "states": np.array(states),
                "actions": np.array(actions),
                "next_states": np.array(next_states)
            }
            trajectories.append(trajectory)
        
        return trajectories

    def compute_importance_weights(self, sampled_trajectory: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute importance sampling weights using equation (10) correctly:
        w(s_t, a_t) = ζ_θ(s_t, a_t) / ζ_θ̄(s_t, a_t)
        
        Where:
        - ζ_θ is the constraint function with current parameters θ
        - ζ_θ̄ is the constraint function with previous parameters θ̄
        """
        states = torch.FloatTensor(sampled_trajectory["states"]).to(self.device)
        actions = torch.FloatTensor(sampled_trajectory["actions"]).to(self.device)
        
        with torch.no_grad():
            # Compute constraint values with current parameters (zeta_theta)
            zeta_theta = self.reward_network(states, actions).squeeze()
            
            # Compute constraint values with previous parameters (zeta_theta_bar)
            zeta_theta_bar = self.prev_reward_network(states, actions).squeeze()
            
            # Ensure values are strictly between 0 and 1 to avoid extreme ratios
            zeta_theta = torch.clamp(zeta_theta, 1e-6, 1.0 - 1e-6)
            zeta_theta_bar = torch.clamp(zeta_theta_bar, 1e-6, 1.0 - 1e-6)
            # Log some values for debugging
            # print(f"Zeta theta stats: min={zeta_theta.min().item():.6f}, max={zeta_theta.max().item():.6f}, mean={zeta_theta.mean().item():.6f}")
            # print(f"Zeta theta bar stats: min={zeta_theta_bar.min().item():.6f}, max={zeta_theta_bar.max().item():.6f}, mean={zeta_theta_bar.mean().item():.6f}")


        # Use log-space calculations to compute trajectory weight
        # This helps avoid numerical overflow from multiplying many weights
        log_weights = []
        for zeta, zeta_bar in zip(zeta_theta, zeta_theta_bar):
            # w(s_t, a_t) = zeta_theta / zeta_theta_bar
            # log(w(s_t, a_t)) = log(zeta_theta) - log(zeta_theta_bar)
            log_weight = np.log(zeta.item()) - np.log(zeta_bar.item())
            
            # Clip to avoid extreme values
            log_weight = np.clip(log_weight, -10.0, 10.0)
            log_weights.append(log_weight)
        
        # Sum the logs instead of multiplying the weights
        log_trajectory_weight = np.sum(log_weights)
        
        # Convert back from log space, with clipping for stability
        trajectory_weight = np.clip(np.exp(log_trajectory_weight), 1e-6, 1e6)
        
        return trajectory_weight
    
    def update_reward_parameters(self, sampled_trajectories: List[Dict[str, np.ndarray]], 
                             weights: np.ndarray) -> None:
        """
        Update reward parameters using gradient descent based on equation (11):
        
        ∇θL(θ) ≈ (1/N) ∑_{i=1}^N ∑_{s_t,a_t∈τ^(i)} ∇θ log ζθ(s_t, a_t) - 
                (1/M) ∑_{j=1}^M ∑_{ŝ_t,â_t∈τ^(j)} ω(ŝ_t, â_t)∇θ log ζθ(ŝ_t, â_t)
        
        This implements the exact gradient computation from equation (11) in the paper.
        
        Args:
            sampled_trajectories: List of trajectories sampled from current policy
            weights: Importance sampling weights for each sampled trajectory
        """
        # Calculate trajectory weights product for each state-action pair
        # For each trajectory, we need to compute ω(τ) = ∏_{t=1}^{T} ω(s_t, a_t)
        # And then apply this weight to each state-action pair in that trajectory
        
        # Process expert trajectories (first term in equation 11)
        N = len(self.expert_trajectories)
        expert_states_list = []
        expert_actions_list = []
        
        for traj in self.expert_trajectories:
            states = traj["states"]
            actions = traj["actions"]
            
            for s, a in zip(states, actions):
                expert_states_list.append(s)
                
                # Handle action format consistently
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
        
        # Process sampled trajectories (second term in equation 11)
        M = len(sampled_trajectories)
        sampled_states_list = []
        sampled_actions_list = []
        sampled_weights_list = []
        
        for i, traj in enumerate(sampled_trajectories):
            # Get the importance weight for this trajectory
            traj_weight = weights[i]
            
            states = traj["states"]
            actions = traj["actions"]
            
            for s, a in zip(states, actions):
                sampled_states_list.append(s)
                sampled_actions_list.append(a)
                sampled_weights_list.append(traj_weight)
        
        # Convert to tensors
        if expert_states_list:
            expert_states = torch.FloatTensor(np.array(expert_states_list)).to(self.device)
            expert_actions = torch.FloatTensor(np.array(expert_actions_list)).to(self.device)
        else:
            # Handle case where there are no expert trajectories (unlikely but possible)
            expert_states = torch.FloatTensor([]).to(self.device)
            expert_actions = torch.FloatTensor([]).to(self.device)
        
        if sampled_states_list:
            sampled_states = torch.FloatTensor(np.array(sampled_states_list)).to(self.device)
            sampled_actions = torch.FloatTensor(np.array(sampled_actions_list)).to(self.device)
            sampled_weights = torch.FloatTensor(np.array(sampled_weights_list)).to(self.device)
        else:
            # Handle case where there are no sampled trajectories (unlikely but possible)
            sampled_states = torch.FloatTensor([]).to(self.device)
            sampled_actions = torch.FloatTensor([]).to(self.device)
            sampled_weights = torch.FloatTensor([]).to(self.device)
        
        # Calculate gradients in batches to avoid memory issues
        expert_batch_size = min(self.batch_size, len(expert_states)) if len(expert_states) > 0 else 0
        sampled_batch_size = min(self.batch_size, len(sampled_states)) if len(sampled_states) > 0 else 0
        
        n_expert_batches = (len(expert_states) + expert_batch_size - 1) // expert_batch_size if expert_batch_size > 0 else 0
        n_sampled_batches = (len(sampled_states) + sampled_batch_size - 1) // sampled_batch_size if sampled_batch_size > 0 else 0
        
        # Training loop for multiple epochs
        n_epochs = 5
        
        for epoch in range(n_epochs):
            # Shuffle data for this epoch
            if len(expert_states) > 0:
                expert_indices = torch.randperm(len(expert_states))
            if len(sampled_states) > 0:
                sampled_indices = torch.randperm(len(sampled_states))
            
            # Process all batches for this epoch
            for batch_idx in range(max(n_expert_batches, n_sampled_batches)):
                self.reward_optimizer.zero_grad()
                total_loss = 0.0
                
                # Process expert batch if available (first term in equation 11)
                if batch_idx < n_expert_batches and len(expert_states) > 0:
                    start_idx = batch_idx * expert_batch_size
                    end_idx = min(start_idx + expert_batch_size, len(expert_states))
                    
                    # Get batch of expert data
                    batch_expert_states = expert_states[expert_indices[start_idx:end_idx]]
                    batch_expert_actions = expert_actions[expert_indices[start_idx:end_idx]]
                    
                    # Forward pass for expert data
                    expert_rewards = self.reward_network(batch_expert_states, batch_expert_actions)
                    
                    # Compute log ζθ(s_t, a_t) for expert data
                    # Now the reward_network outputs are already in [0, 1] due to sigmoid
                    expert_log_zeta = torch.log(expert_rewards + 1e-8)
                    
                    # First term: (1/N) ∑_{i=1}^N ∑_{s_t,a_t∈τ^(i)} ∇θ log ζθ(s_t, a_t)
                    # Since we're doing gradient descent, we negate this term
                    expert_loss = -torch.sum(expert_log_zeta) / (N + 1e-8)
                    total_loss += expert_loss
                
                # Process sampled batch if available (second term in equation 11)
                if batch_idx < n_sampled_batches and len(sampled_states) > 0:
                    start_idx = batch_idx * sampled_batch_size
                    end_idx = min(start_idx + sampled_batch_size, len(sampled_states))
                    
                    # Get batch of sampled data with weights
                    batch_sampled_states = sampled_states[sampled_indices[start_idx:end_idx]]
                    batch_sampled_actions = sampled_actions[sampled_indices[start_idx:end_idx]]
                    batch_weights = sampled_weights[sampled_indices[start_idx:end_idx]]
                    
                    # Forward pass for sampled data
                    sampled_rewards = self.reward_network(batch_sampled_states, batch_sampled_actions)
                    
                    # Compute log ζθ(ŝ_t, â_t) for sampled data
                    sampled_log_zeta = torch.log(sampled_rewards + 1e-8)
                    
                    # Second term: (1/M) ∑_{j=1}^M ∑_{ŝ_t,â_t∈τ^(j)} ω(ŝ_t, â_t)∇θ log ζθ(ŝ_t, â_t)
                    # We multiply each log_zeta by its importance weight
                    # Since we're doing gradient descent, we include this term with positive sign
                    sampled_loss = torch.sum(batch_weights.unsqueeze(1) * sampled_log_zeta) / (M + 1e-8)
                    total_loss += sampled_loss
                
                # Backward pass and optimization step
                total_loss.backward()
                # Before optimizer step
                torch.nn.utils.clip_grad_norm_(self.reward_network.parameters(), max_norm=1.0)
                self.reward_optimizer.step()
                
        # Optional: Add regularization to prevent overfitting
        # This is not in the original equation but can help with stability
        l2_reg = 0.01
        for param in self.reward_network.parameters():
            total_loss += l2_reg * torch.sum(param ** 2)

    def compute_kl_divergences(self, sampled_trajectories: List[Dict[str, np.ndarray]],
                           weights: np.ndarray) -> Tuple[float, float]:
        """
        Compute forward and reverse KL divergences using equation (12):
        D_KL(π_θ||π_θ̃) ≤ 2 log(w̄)
        D_KL(π_θ̃||π_θ) ≤ E_{τ~π_θ̃}[(ω(τ) - w̄) log(ω(τ))] / w̄
        
        where w̄ = E_{τ~π_θ̃}[ω(τ)] and ω(τ) = ∏_{t=1}^{T} ω(s_t, a_t)
        
        Args:
            sampled_trajectories: List of trajectories sampled from current policy
            weights: Importance sampling weights for each trajectory
            
        Returns:
            Tuple containing (forward_kl, reverse_kl)
        """
        # Compute w̄ = E_{τ~π_θ̃}[ω(τ)]
        mean_weight = np.mean(weights) if len(weights) > 0 else 1e-8
        
        # Compute forward KL: D_KL(π_θ||π_θ̃) ≤ 2 log(w̄)
        # Add small constant to avoid log(0)
        forward_kl = 2 * np.log(mean_weight + 1e-8)
        
        # Compute terms for reverse KL
        log_weights = np.log(weights + 1e-8)
        weighted_log_weights = (weights - mean_weight) * log_weights
        
        # Compute reverse KL: D_KL(π_θ̃||π_θ) ≤ E_{τ~π_θ̃}[(ω(τ) - w̄) log(ω(τ))] / w̄
        reverse_kl = np.sum(weighted_log_weights) / (mean_weight + 1e-8)
        
        return forward_kl, reverse_kl

    def train(self, n_iterations: int, n_trajectories_per_iter: int, reward_threshold: float = 400, eval_episodes: int = 10) -> Dict:
        """
        Main training loop for ICRL algorithm as described in Algorithm 1.
        
        Args:
            n_iterations: Number of outer iterations (N in Algorithm 1)
            n_trajectories_per_iter: Number of trajectories to sample (M in Algorithm 1)
            
        Returns:
            Dictionary with training statistics and learned policy
        """
        stats = {
            'forward_kls': [],
            'reverse_kls': [],
            'eval_rewards': []  # Track evaluation rewards
        }
        early_stop = False
        
        print("Starting ICRL training...")
        
        for i in range(n_iterations):
            print(f"Outer iteration {i+1}/{n_iterations}")
            
            # Learn policy π^φ by solving equation (9)
            print("Learning policy...")
            policy_result = self.learn_policy()
            current_policy = policy_result["policy"]
            # Evaluate current policy
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
        
            # Early stopping check based on reward
            if avg_reward >= reward_threshold:
                print(f"  Reached reward threshold of {reward_threshold}. Early stopping.")
                early_stop = True
                break
            
            for j in range(self.backward_iterations):
                print(f"  Backward iteration {j+1}/{self.backward_iterations}")
                
                # Before updating θ, copy current θ to θ̄
                self.prev_reward_network.load_state_dict(self.reward_network.state_dict())

                # For debugging, get a parameter norm before update
                before_update_norm = sum(p.norm().item() for p in self.reward_network.parameters())

                
                # Sample set of trajectories from the current policy
                print("  Sampling trajectories...")
                sampled_trajectories = self.sample_trajectories(current_policy, n_trajectories_per_iter)
                
                # Compute importance sampling weights using equation (10)
                print("  Computing importance weights...")
                weights = np.array([self.compute_importance_weights(traj) for traj in sampled_trajectories])
                
                # Monitor weights for extreme values
                if not np.all(np.isfinite(weights)):
                    non_finite_count = np.sum(~np.isfinite(weights))
                    print(f"Warning: {non_finite_count} non-finite weight values detected")
                    weights = np.array([w if np.isfinite(w) else 1.0 for w in weights])
                
                # Additional logging
                print(f"  Weight stats - min: {np.min(weights):.6e}, max: {np.max(weights):.6e}, mean: {np.mean(weights):.6e}")
                
                # Update θ via SGD using equation (11)
                print("  Updating reward parameters...")
                self.update_reward_parameters(sampled_trajectories, weights)
                # Get parameter norm after updat
                after_update_norm = sum(p.norm().item() for p in self.reward_network.parameters())
                print(f"  Parameter norm before update: {before_update_norm:.6f}, after update: {after_update_norm:.6f}")

                
                # Compute forward and reverse KL divergences using equation (12)
                forward_kl, reverse_kl = self.compute_kl_divergences(sampled_trajectories, weights)
                stats['forward_kls'].append(forward_kl)
                stats['reverse_kls'].append(reverse_kl)
                
                print(f"  Forward KL: {forward_kl:.6f}, Reverse KL: {reverse_kl:.6f}")
                
                # Check convergence conditions
                if forward_kl >= self.forward_kl_threshold or reverse_kl >= self.reverse_kl_threshold:
                    print(f"  Converged at backward iteration {j+1}!")
                    break

                if forward_kl == 0 and reverse_kl == 0:
                    print(f"  Converged at backward iteration {j+1}!")
                    break
        
        # Final policy learning
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


def process_expert_demonstrations(demos: List[Dict]) -> List[Dict[str, np.ndarray]]:
    """
    Process expert demonstrations into the format expected by ICRL.
    
    Args:
        demos: List of expert demonstrations, each containing states, actions, etc.
        
    Returns:
        Processed trajectories in ICRL format
    """
    processed_trajectories = []
    
    for demo in demos:
        # Check if the demonstration has the required keys
        if "observations" in demo and "actions" in demo:
            states = np.array(demo["observations"][:-1])
            actions = np.array(demo["actions"])
            next_states = np.array(demo["observations"][1:])
            
            # Ensure states and actions have the same length
            min_len = min(len(states), len(actions), len(next_states))
            
            processed_trajectory = {
                "states": states[:min_len],
                "actions": actions[:min_len],
                "next_states": next_states[:min_len]
            }
            processed_trajectories.append(processed_trajectory)
    
    return processed_trajectories



def example_usage():
    """Example of how to use the ICRL implementation with Gymnasium."""
    # Print current random states
    # np.random.seed(2147483648)
    # torch.manual_seed(2339084767257311861)
    print(f"Current PyTorch seed: {torch.initial_seed()}")
    print(f"Current NumPy RNG state: {np.random.get_state()[1][0]}")  # First element of the state array
    
    # Create a Gymnasium environment
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    
    # Generate some expert demonstrations (in a real scenario, these would come from an expert)
    # For this example, we'll use a simple heuristic as our "expert"
    if env_name == 'CartPole-v1':
        def expert_policy(state):
            """Simple expert policy for CartPole."""
            if state[2] < 0:  # If pole is falling to the left
                return 0  # Move cart to the left
            else:
                return 1  # Move cart to the right
    elif env_name == 'DummyGrid-v0':
        def expert_policy(state):
            """Simple expert policy for SimpleGrid."""
            # 0 = up, 1 = down, 2 = left, 3 = right
            position = np.argmax(state)
            if position in [0, 1, 2, 3, 4, 5 ]:
                return 1 # down
            else:
                return 3 # right
    
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
    
    # Create and train ICRL
    icrl = ICRL(
        env=env,
        expert_trajectories=expert_trajectories,
        backward_iterations=2,
        forward_kl_threshold=0.01,
        reverse_kl_threshold=0.01
    )
    
    # Train ICRL
    results = icrl.train(n_iterations=1000, n_trajectories_per_iter=100)
    
    # Save the trained model
    icrl.save("icrl_cartpole.pt")
    
    # Evaluate the learned policy
    learned_policy = results["policy"]
    eval_rewards = []
    
    for _ in range(10):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = learned_policy.sample_action(state)
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
            action = learned_policy.sample_greedy_action(learned_policy, state)
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