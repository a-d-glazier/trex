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
    
    def forward(self, states, actions):
        """Forward pass through the reward network."""
        x = torch.cat([states, actions], dim=1)
        return self.network(x)


class ConstraintNetwork(nn.Module):
    """Neural network to represent the constraint function."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output is between 0 and 1 for constraint satisfaction probability
        )
    
    def forward(self, states, actions):
        """Forward pass through the constraint network."""
        x = torch.cat([states, actions], dim=1)
        return self.network(x)


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
        forward_kl_threshold: float = 0.01,
        reverse_kl_threshold: float = 0.01,
        reward_lr: float = 0.001,
        constraint_lr: float = 0.001,
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
        
        # Initialize reward and constraint networks
        self.reward_network = RewardNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.constraint_network = ConstraintNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        
        # Setup optimizers
        self.reward_optimizer = optim.Adam(self.reward_network.parameters(), lr=reward_lr)
        self.constraint_optimizer = optim.Adam(self.constraint_network.parameters(), lr=constraint_lr)
        
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
        Learn a policy by solving equation (9) in the paper.
        For compatibility with Gymnasium, we'll use a simple policy gradient approach.
        Returns the learned policy.
        """
        # For simplicity, we'll implement a basic policy gradient with constraints
        # In a real implementation, you might want to use more sophisticated RL methods
        
        class Policy(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim, discrete):
                super().__init__()
                self.discrete = discrete
                
                self.network = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
                
                if discrete:
                    self.action_head = nn.Linear(hidden_dim, action_dim)
                else:
                    self.mean_head = nn.Linear(hidden_dim, action_dim)
                    self.log_std_head = nn.Linear(hidden_dim, action_dim)
                    self.log_std_min = -20
                    self.log_std_max = 2
            
            def forward(self, state):
                x = self.network(state)
                
                if self.discrete:
                    action_probs = torch.softmax(self.action_head(x), dim=-1)
                    return action_probs
                else:
                    mean = self.mean_head(x)
                    log_std = torch.clamp(self.log_std_head(x), self.log_std_min, self.log_std_max)
                    return mean, torch.exp(log_std)
            
            def sample_action(self, state):
                state = torch.FloatTensor(state).unsqueeze(0)
                
                if self.discrete:
                    probs = self.forward(state)
                    action = torch.multinomial(probs, 1).item()
                    return action
                else:
                    mean, std = self.forward(state)
                    normal = torch.distributions.Normal(mean, std)
                    action = normal.sample()
                    return action.detach().numpy().squeeze()
        
        # Initialize policy
        if self.discrete_actions:
            action_dim = self.n_actions
        else:
            action_dim = self.action_dim
            
        policy = Policy(self.state_dim, action_dim, 128, self.discrete_actions).to(self.device)
        policy_optimizer = optim.Adam(policy.parameters(), lr=0.001)
        
        # Training loop for policy learning
        max_episodes = 1000
        max_steps = 1000
        
        for episode in range(max_episodes):
            state, _ = self.env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_constraints = []
            
            for step in range(max_steps):
                # Sample action from policy
                with torch.no_grad():
                    if self.discrete_actions:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        probs = policy(state_tensor)
                        action = torch.multinomial(probs, 1).item()
                    else:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        mean, std = policy(state_tensor)
                        normal = torch.distributions.Normal(mean, std)
                        action = normal.sample().cpu().numpy().squeeze()
                
                # Execute action in environment
                next_state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                episode_states.append(state)
                
                if self.discrete_actions:
                    action_tensor = torch.zeros(self.n_actions)
                    action_tensor[action] = 1
                    episode_actions.append(action_tensor.numpy())
                else:
                    episode_actions.append(action)
                
                # Compute reward and constraint values
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                if self.discrete_actions:
                    action_tensor = torch.zeros(1, self.n_actions).to(self.device)
                    action_tensor[0, action] = 1
                else:
                    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    reward = self.reward_network(state_tensor, action_tensor).item()
                    constraint = self.constraint_network(state_tensor, action_tensor).item()
                
                episode_rewards.append(reward)
                episode_constraints.append(constraint)
                
                state = next_state
                
                if done:
                    break
            
            # Compute returns
            returns = []
            G = 0
            for reward in reversed(episode_rewards):
                G = reward + self.gamma * G
                returns.insert(0, G)
            returns = torch.FloatTensor(returns).to(self.device)
            
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Convert episode data to tensors
            states = torch.FloatTensor(np.array(episode_states)).to(self.device)
            if self.discrete_actions:
                actions = torch.FloatTensor(np.array(episode_actions)).to(self.device)
                action_probs = policy(states)
                log_probs = torch.log(torch.sum(action_probs * actions, dim=1) + 1e-8)
            else:
                actions = torch.FloatTensor(np.array(episode_actions)).to(self.device)
                mean, std = policy(states)
                normal = torch.distributions.Normal(mean, std)
                log_probs = normal.log_prob(actions).sum(dim=1)
            
            # Compute constraint penalty
            constraints = torch.FloatTensor(episode_constraints).to(self.device)
            constraint_penalty = -torch.log(constraints + 1e-8).mean()
            
            # Compute policy loss
            policy_loss = -(log_probs * returns).mean() + constraint_penalty
            
            # Update policy
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()
            
            # Check if constraints are satisfied
            if episode > 10 and constraint_penalty.item() < 0.1:
                print(f"Policy converged at episode {episode}")
                break

            # break if constraint penalty isn't changing
            if episode > 1 and abs(constraint_penalty.item() - constraint_penalty_prev) < 0.001:
                print(f"Constraint penalty converged at episode {episode}")
                break
            constraint_penalty_prev = constraint_penalty.item()
        
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
        """Compute importance sampling weights for a sampled trajectory using equation (10)."""
        states = torch.FloatTensor(sampled_trajectory["states"]).to(self.device)
        actions = torch.FloatTensor(sampled_trajectory["actions"]).to(self.device)
        
        with torch.no_grad():
            # Compute rewards
            rewards = self.reward_network(states, actions).squeeze()
            
            # Compute constraints
            constraints = self.constraint_network(states, actions).squeeze()
        
        # Compute importance weights as in equation (10)
        # w(τ) ∝ exp(r(τ)) if c(τ) ≥ 0, else 0
        rewards_sum = rewards.sum().item()
        constraints_min = constraints.min().item()
        
        # Check if trajectory satisfies constraints
        if constraints_min >= 0.5:  # Threshold for constraint satisfaction
            weight = np.exp(rewards_sum)
        else:
            weight = 0.0
        
        return weight
    
    def update_constraint_parameters(self, sampled_trajectories: List[Dict[str, np.ndarray]], 
                                     weights: np.ndarray) -> Tuple[float, float]:
        """Update constraint parameters using SGD with equation (11) and (12)."""
        # Normalize weights for importance sampling
        if np.sum(weights) > 0:
            normalized_weights = weights / np.sum(weights)
        else:
            # If all trajectories have zero weight, use uniform weights
            normalized_weights = np.ones_like(weights) / len(weights)
        
        # Flatten all trajectories into single arrays
        all_states = []
        all_actions = []
        all_labels = []  # 1 for expert, 0 for sampled
        
        # Expert trajectories
        for traj in self.expert_trajectories:
            states = traj["states"]
            actions = traj["actions"]
            
            # Add each state-action pair individually
            for s, a in zip(states, actions):
                all_states.append(s)

                # Ensure consistent action format
                if self.discrete_actions:
                    # Convert to one-hot if needed
                    if np.isscalar(a) or (isinstance(a, np.ndarray) and a.size == 1):
                        a_idx = int(a) if np.isscalar(a) else int(a.item())
                        a_onehot = np.zeros(self.n_actions)
                        a_onehot[a_idx] = 1
                        all_actions.append(a_onehot)
                    else:
                        # Already in correct format (e.g., one-hot)
                        all_actions.append(a)
                else:
                    # For continuous actions, ensure correct shape
                    all_actions.append(np.array(a).flatten())
                all_labels.append(1.0)  # Expert label
        
        # Sampled trajectories with importance weights
        for i, traj in enumerate(sampled_trajectories):
            states = traj["states"]
            actions = traj["actions"]
            weight = normalized_weights[i]
            
            # Add each state-action pair individually
            for s, a in zip(states, actions):
                all_states.append(s)
                all_actions.append(a)
                all_labels.append(weight)  # Weight as label
        
        # Convert to tensors
        all_states = torch.FloatTensor(np.array(all_states)).to(self.device)
        all_actions = torch.FloatTensor(np.array(all_actions)).to(self.device)
        all_labels = torch.FloatTensor(np.array(all_labels)).to(self.device)
            
        # Shuffle data
        indices = torch.randperm(len(all_states))
        all_states = all_states[indices]
        all_actions = all_actions[indices]
        all_labels = all_labels[indices]
        
        # Training loop for constraint network
        n_epochs = 5
        batch_size = min(self.batch_size, len(all_states))
        
        # Track KL divergences
        forward_kls = []
        reverse_kls = []
        
        for _ in range(n_epochs):
            for i in range(0, len(all_states), batch_size):
                batch_states = all_states[i:i+batch_size]
                batch_actions = all_actions[i:i+batch_size]
                batch_labels = all_labels[i:i+batch_size]
                
                # Forward pass
                constraint_preds = self.constraint_network(batch_states, batch_actions).squeeze()
                
                # Compute binary cross-entropy loss for constraint
                expert_mask = (batch_labels == 1)
                sampled_mask = (batch_labels < 1) & (batch_labels > 0)  # Exclude zero-weight samples
                
                # Compute KL divergences if we have both expert and sampled data
                if expert_mask.sum() > 0 and sampled_mask.sum() > 0:
                    # Forward KL: Expert || Sampled
                    p_expert = constraint_preds[expert_mask].mean()
                    p_sampled = constraint_preds[sampled_mask].mean()
                    forward_kl = p_expert * torch.log(p_expert / p_sampled) + (1 - p_expert) * torch.log((1 - p_expert) / (1 - p_sampled))
                    
                    # Reverse KL: Sampled || Expert
                    reverse_kl = p_sampled * torch.log(p_sampled / p_expert) + (1 - p_sampled) * torch.log((1 - p_sampled) / (1 - p_expert))
                    
                    forward_kls.append(forward_kl.item())
                    reverse_kls.append(reverse_kl.item())
                
                # Binary cross-entropy loss with weights
                bce_loss = -batch_labels * torch.log(constraint_preds + 1e-8) - (1 - batch_labels) * torch.log(1 - constraint_preds + 1e-8)
                loss = bce_loss.mean()
                
                # Update constraint network
                self.constraint_optimizer.zero_grad()
                loss.backward()
                self.constraint_optimizer.step()
        
        # Compute average KL divergences
        forward_kl = np.mean(forward_kls) if forward_kls else float('inf')
        reverse_kl = np.mean(reverse_kls) if reverse_kls else float('inf')
        
        return forward_kl, reverse_kl
    
    def update_reward_parameters(self, sampled_trajectories: List[Dict[str, np.ndarray]], 
                                  weights: np.ndarray) -> None:
        """Update constraint parameters using SGD with equation (11) and (12)."""
        # Normalize weights for importance sampling
        if np.sum(weights) > 0:
            normalized_weights = weights / np.sum(weights)
        else:
            # If all trajectories have zero weight, use uniform weights
            normalized_weights = np.ones_like(weights) / len(weights)
        
        # Flatten all trajectories into single arrays
        all_states = []
        all_actions = []
        all_labels = []  # 1 for expert, 0 for sampled
        
        # Expert trajectories
        for traj in self.expert_trajectories:
            states = traj["states"]
            actions = traj["actions"]
            
            # Add each state-action pair individually with consistent action format
            for s, a in zip(states, actions):
                all_states.append(s)
                
                # Ensure consistent action format
                if self.discrete_actions:
                    # Convert to one-hot if needed
                    if np.isscalar(a) or (isinstance(a, np.ndarray) and a.size == 1):
                        a_idx = int(a) if np.isscalar(a) else int(a.item())
                        a_onehot = np.zeros(self.n_actions)
                        a_onehot[a_idx] = 1
                        all_actions.append(a_onehot)
                    else:
                        # Already in correct format (e.g., one-hot)
                        all_actions.append(a)
                else:
                    # For continuous actions, ensure correct shape
                    all_actions.append(np.array(a).flatten())
                
                all_labels.append(1.0)  # Expert label
        
        # Sampled trajectories with importance weights
        for i, traj in enumerate(sampled_trajectories):
            states = traj["states"]
            actions = traj["actions"]
            weight = normalized_weights[i]
            
            # Add each state-action pair individually with consistent action format
            for s, a in zip(states, actions):
                all_states.append(s)
                
                # Ensure consistent action format
                if self.discrete_actions:
                    # Convert to one-hot if needed
                    if np.isscalar(a) or (isinstance(a, np.ndarray) and a.size == 1):
                        a_idx = int(a) if np.isscalar(a) else int(a.item())
                        a_onehot = np.zeros(self.n_actions)
                        a_onehot[a_idx] = 1
                        all_actions.append(a_onehot)
                    else:
                        # Already in correct format (e.g., one-hot)
                        all_actions.append(a)
                else:
                    # For continuous actions, ensure correct shape
                    all_actions.append(np.array(a).flatten())
                
                all_labels.append(weight)  # Weight as label
        
        # Convert to tensors
        all_states = torch.FloatTensor(np.array(all_states)).to(self.device)
        all_actions = torch.FloatTensor(np.array(all_actions)).to(self.device)
        all_labels = torch.FloatTensor(np.array(all_labels)).to(self.device)
            
        # Shuffle data
        indices = torch.randperm(len(all_states))
        all_states = all_states[indices]
        all_actions = all_actions[indices]
        all_labels = all_labels[indices]
        
        # Training loop for reward network
        n_epochs = 5
        batch_size = min(self.batch_size, len(all_states))
        
        for _ in range(n_epochs):
            for i in range(0, len(all_states), batch_size):
                batch_states = all_states[i:i+batch_size]
                batch_actions = all_actions[i:i+batch_size]
                batch_labels = all_labels[i:i+batch_size]
                
                # Forward pass
                reward_preds = self.reward_network(batch_states, batch_actions).squeeze()
                
                # Weighted MSE loss: weight * (reward - target)^2
                # Higher weight for expert data
                loss = batch_labels * (reward_preds - 1.0)**2 + (1 - batch_labels) * reward_preds**2
                loss = loss.mean()
                
                # Update reward network
                self.reward_optimizer.zero_grad()
                loss.backward()
                self.reward_optimizer.step()
    
    def train(self, n_iterations: int, n_trajectories_per_iter: int) -> Dict:
        """
        Main training loop for ICRL algorithm.
        
        Args:
            n_iterations: Number of iterations to run the algorithm
            n_trajectories_per_iter: Number of trajectories to sample in each iteration
            
        Returns:
            Dictionary with training statistics and learned policy
        """
        stats = {
            'forward_kls': [],
            'reverse_kls': []
        }
        
        # Initialize policy and constraint parameters randomly
        print("Starting ICRL training...")
        
        for iteration in range(n_iterations):
            print(f"Iteration {iteration+1}/{n_iterations}")
            
            # Learn policy π* by solving equation (9)
            print("Learning policy...")
            policy_result = self.learn_policy()
            current_policy = policy_result["policy"]
            
            # Sample set of trajectories from the current policy
            print("Sampling trajectories...")
            sampled_trajectories = self.sample_trajectories(current_policy, n_trajectories_per_iter)
            
            # Compute importance sampling weights
            print("Computing importance weights...")
            weights = np.array([self.compute_importance_weights(traj) for traj in sampled_trajectories])
            
            # Update constraints via SGD using equation (11)
            print("Updating constraint parameters...")
            forward_kl, reverse_kl = self.update_constraint_parameters(sampled_trajectories, weights)
            stats['forward_kls'].append(forward_kl)
            stats['reverse_kls'].append(reverse_kl)
            
            # Update reward parameters
            print("Updating reward parameters...")
            self.update_reward_parameters(sampled_trajectories, weights)
            
            print(f"Forward KL: {forward_kl:.6f}, Reverse KL: {reverse_kl:.6f}")
            
            # Check convergence
            # if forward_kl <= self.forward_kl_threshold and reverse_kl <= self.reverse_kl_threshold:
            #     print(f"Converged at iteration {iteration+1}!")
            #     break
        
        # Final policy learning
        final_policy_result = self.learn_policy()
        self.policy = final_policy_result["policy"]
        
        return {
            "policy": self.policy,
            "reward_network": self.reward_network,
            "constraint_network": self.constraint_network,
            "stats": stats
        }
    
    def save(self, path: str) -> None:
        """Save the trained models."""
        torch.save({
            "reward_network": self.reward_network.state_dict(),
            "constraint_network": self.constraint_network.state_dict(),
            "policy": self.policy.state_dict() if self.policy else None
        }, path)
    
    def load(self, path: str) -> None:
        """Load trained models."""
        checkpoint = torch.load(path)
        self.reward_network.load_state_dict(checkpoint["reward_network"])
        self.constraint_network.load_state_dict(checkpoint["constraint_network"])
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
    torch.manual_seed(2339084767257311861)
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
    n_expert_demos = 10
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
        backward_iterations=10,
        forward_kl_threshold=0.01,
        reverse_kl_threshold=0.01
    )
    
    # Train ICRL
    results = icrl.train(n_iterations=10, n_trajectories_per_iter=5)
    
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
            action = learned_policy.sample_action(state)
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