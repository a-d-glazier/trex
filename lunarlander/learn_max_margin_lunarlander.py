# -*- coding: utf-8 -*-
"""
MaxMargin IRL with PPO Expert for LunarLander-v3
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
from typing import List, Tuple, Dict, Optional, Union, Any
import random
import os
import argparse
import time
import matplotlib.pyplot as plt # Added for plotting results
import pickle # Used for saving/loading demos if needed, though PPO model is primary

# --- Feature Engineering ---
def get_state_features(state):
    """
    Enhanced features for LunarLander, focusing on critical aspects for landing.
    """
    state = np.asarray(state, dtype=np.float32)
    if state.shape[0] != 8:
        raise ValueError(f"Expected state dim 8, got {state.shape}")

    x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = state

    # Enhanced feature set
    features = [
        1.0,                    # Bias
        y,                      # Vertical position
        vy,                     # Vertical velocity
        angle,                  # Angle
        angular_velocity,       # Angular velocity
        left_leg,              # Left leg contact
        right_leg,             # Right leg contact
        np.abs(x),             # Absolute horizontal position
        np.abs(vx),            # Absolute horizontal velocity
        np.abs(vy),            # Absolute vertical velocity
        np.abs(angle),         # Absolute angle
        np.abs(angular_velocity), # Absolute angular velocity
        left_leg * right_leg,  # Both legs contact
        np.exp(-np.abs(y)),    # Height-based reward shaping
        np.exp(-np.abs(angle)) # Angle-based reward shaping
    ]
    return np.array(features, dtype=np.float32)

# --- Policy Network (for IRL Agent) ---
class PolicyNetwork(nn.Module):
    """Policy network for LunarLander."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # Output logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits

# --- MaxMargin IRL Class (with consistent feature normalization) ---
class MaxMarginIRL:
    """
    MaxMargin IRL using projection method, with consistent feature normalization.
    """
    def __init__(
        self,
        env: gym.Env,
        expert_trajectories: List[Dict[str, np.ndarray]],
        gamma: float = 0.99,
        # --- Hyperparameters for Internal RL (Policy Optimization) ---
        learning_rate_policy: float = 1e-4, # Lower LR might be needed for stability
        n_policy_epochs: int = 25,          # Increased epochs for better convergence per iter
        policy_batch_size: int = 128,       # Larger batch size
        # --- IRL Algorithm Parameters ---
        max_previous_mu: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.env = env
        self.expert_trajectories = self._filter_valid_trajectories(expert_trajectories)
        if not self.expert_trajectories:
             raise ValueError("No valid expert trajectories provided after filtering.")

        self.gamma = gamma
        self.device = device
        # Store RL hyperparameters
        self.lr_policy = learning_rate_policy
        self.n_policy_epochs = n_policy_epochs
        self.policy_batch_size = policy_batch_size
        print(f"Using device: {self.device}")
        print(f"Policy Opt Params: LR={self.lr_policy}, Epochs={self.n_policy_epochs}, Batch={self.policy_batch_size}")

        # State/Action Dimensions
        self.state_dim = env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = env.action_space.n
            self.n_actions = env.action_space.n
        else:
            raise ValueError("Only discrete action spaces are supported")

        # Feature Dimension (using the updated get_state_features)
        _dummy_state = env.observation_space.sample()
        self.n_features = len(get_state_features(_dummy_state))
        print(f"Using {self.n_features} features (simplified).")

        # Policy Network & Optimizer
        self.policy = PolicyNetwork(self.state_dim, self.n_actions).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr_policy)

        # Calculate State & Feature Normalization Stats from Expert Data
        all_states, all_features = [], []
        for traj in self.expert_trajectories:
            valid_states = traj["states"]
            all_states.extend(valid_states)
            for state in valid_states:
                 try:
                     all_features.append(get_state_features(state))
                 except ValueError:
                     # print(f"Warning: Skipping state during feature extraction for normalization: {state}") # Can be verbose
                     continue # Skip if features fail

        if not all_states:
            raise ValueError("No valid states found in filtered expert trajectories for normalization.")

        all_states = np.array(all_states, dtype=np.float32)
        all_features = np.array(all_features, dtype=np.float32)

        if all_features.shape[0] == 0:
            raise ValueError("No valid features generated from expert trajectories for normalization.")

        self.state_mean = np.mean(all_states, axis=0)
        self.state_std = np.std(all_states, axis=0) + 1e-8
        self.feature_mean = np.mean(all_features, axis=0)
        self.feature_std = np.std(all_features, axis=0) + 1e-8
        self.feature_std[self.feature_std < 1e-8] = 1.0 # Prevent division by zero

        print("Feature Mean:", self.feature_mean)
        print("Feature Std:", self.feature_std)

        # --- Expert Feature Expectations (NOW uses NORMALIZED features) ---
        print("Computing expert feature expectations (using normalized features)...")
        # This call now uses the MODIFIED compute_feature_expectations below
        self.expert_feature_expectations = self.compute_feature_expectations(self.expert_trajectories)
        print(f"Expert Feature Expectations (normalized): {self.expert_feature_expectations.cpu().numpy()}")
        if torch.isnan(self.expert_feature_expectations).any() or torch.isinf(self.expert_feature_expectations).any():
            raise ValueError("Expert feature expectations contain NaN/Inf after calculation.")

        # Reward Weights & History
        self.reward_weights = torch.zeros(self.n_features, device=self.device)
        self.previous_feature_expectations = deque(maxlen=max_previous_mu)

        # --- Initialization with random policy MU (NOW uses NORMALIZED features) ---
        print("Initializing with random policy feature expectations (using normalized features)...")
        initial_trajectories = self.sample_trajectories(10, use_current_policy=False)
        if not initial_trajectories:
             print("Warning: Failed to sample initial trajectories. Using zero vector for initial mu.")
             initial_mu = torch.zeros(self.n_features, device=self.device)
        else:
            # This call also uses the MODIFIED compute_feature_expectations
            initial_mu = self.compute_feature_expectations(initial_trajectories)
            if torch.isnan(initial_mu).any() or torch.isinf(initial_mu).any():
                 print("Warning: Initial random MU contains NaN/Inf. Using zero vector.")
                 initial_mu = torch.zeros(self.n_features, device=self.device)

        self.previous_feature_expectations.append(initial_mu.to(self.device))

        # Initial Weights (based on NORMALIZED expectations)
        self.reward_weights = self.expert_feature_expectations.to(self.device) - initial_mu.to(self.device)
        norm = torch.norm(self.reward_weights)
        if norm > 1e-8:
            self.reward_weights /= norm
        else:
            # If expert mu == initial mu, weights are zero. Start with small random weights.
             print("Warning: Initial weights norm is zero. Using small random weights.")
             self.reward_weights = torch.randn(self.n_features, device=device) * 0.01

        print(f"Initial Reward Weights (normalized): {self.reward_weights.cpu().numpy()}")

        # Exploration
        self.exploration_rate = 0.3  # Start with moderate exploration
        self.min_exploration_rate = 0.01  # Lower minimum exploration
        self.exploration_decay = 0.98  # Decay faster

    def _filter_valid_trajectories(self, trajectories):
        """Ensure trajectories have valid states matching env observation space."""
        filtered = []
        if not trajectories: return filtered
        try: expected_dim = self.env.observation_space.shape[0]
        except AttributeError: expected_dim = 8 # Fallback for LunarLander
        for i, traj in enumerate(trajectories):
            states = None
            # Try to get states s0..s(T-1)
            if "states" in traj and isinstance(traj["states"], np.ndarray):
                 states = traj["states"]
            elif "observations" in traj and isinstance(traj["observations"], np.ndarray):
                 # Infer states s0..s(T-1) from observations s0..sT
                 if len(traj["observations"]) > 1 and len(traj["observations"]) == len(traj.get("actions", [])) + 1:
                     states = traj["observations"][:-1]
                 else:
                     # print(f"Skipping traj {i}: Obs length {len(traj['observations'])} mismatch with action length {len(traj.get('actions', []))}")
                     continue
            else:
                # print(f"Skipping traj {i}: Missing 'states' or 'observations' or invalid types.")
                continue

            # Validate the inferred/provided states
            if states is None or not isinstance(states, np.ndarray) or states.ndim != 2 or states.shape[1] != expected_dim:
                 # print(f"Skipping traj {i}: Invalid states shape {states.shape if states is not None else 'None'} or dim {states.shape[1] if states is not None and states.ndim == 2 else 'N/A'} vs expected {expected_dim}")
                 continue
            if np.isnan(states).any() or np.isinf(states).any():
                 # print(f"Skipping traj {i}: NaN/Inf found in states.")
                 continue

            # If we got here, states are valid s0..s(T-1)
            # Ensure actions are also present and valid (length T)
            actions = traj.get("actions")
            if actions is None or not isinstance(actions, np.ndarray) or len(actions) != len(states):
                 # print(f"Skipping traj {i}: Invalid or missing actions (len {len(actions) if actions is not None else 'None'} vs states len {len(states)})")
                 continue

            # Add the validated parts to the filtered list
            valid_traj = {
                "states": np.asarray(states, dtype=np.float32),
                "actions": np.asarray(actions, dtype=np.int64)
                # Optionally add other keys if needed later, like 'observations'
            }
            if "observations" in traj: valid_traj["observations"] = traj["observations"]

            filtered.append(valid_traj)

        print(f"Filtered {len(trajectories)} input trajectories down to {len(filtered)} valid ones.")
        return filtered


    def normalize_state(self, state):
        """Normalize state using precomputed statistics."""
        state = np.asarray(state, dtype=np.float32)
        norm_state = (state - self.state_mean) / self.state_std
        # Check for NaN/Inf after normalization
        if np.isnan(norm_state).any() or np.isinf(norm_state).any():
             # Handle NaN/Inf, e.g., by clipping or returning a default
             # print(f"Warning: NaN/Inf in normalized state: {norm_state}. Clipping.") # Can be verbose
             norm_state = np.nan_to_num(norm_state, nan=0.0, posinf=1e6, neginf=-1e6) # Replace with 0 or clip
        return norm_state


    def normalize_features(self, features):
        """Normalize features using precomputed statistics."""
        features = np.asarray(features, dtype=np.float32)
        norm_features = (features - self.feature_mean) / self.feature_std
        # Check for NaN/Inf after normalization
        if np.isnan(norm_features).any() or np.isinf(norm_features).any():
             # Handle NaN/Inf
             # print(f"Warning: NaN/Inf in normalized features: {norm_features}. Clipping.") # Can be verbose
             norm_features = np.nan_to_num(norm_features, nan=0.0, posinf=1e6, neginf=-1e6)
        return norm_features


    def compute_reward(self, state: np.ndarray) -> float:
        """Compute reward using normalized features and current weights."""
        try:
            features = get_state_features(state)
            normalized_features = self.normalize_features(features)
            features_tensor = torch.FloatTensor(normalized_features).to(self.device)

            # Check for NaNs/Infs in normalized features before dot product
            if torch.isnan(features_tensor).any() or torch.isinf(features_tensor).any():
                # Return 0 reward if features are invalid
                return 0.0

            reward = torch.dot(features_tensor, self.reward_weights)

            # Check for NaNs/Infs in the final reward
            if torch.isnan(reward).any() or torch.isinf(reward).any():
                 # Return 0 if the reward calculation failed
                 return 0.0

            # Optional: Clip reward to prevent extreme values destabilizing learning
            # reward = torch.clamp(reward, -10.0, 10.0)

            return reward.item()
        except ValueError: # Catch potential get_state_features errors
             return 0.0
        except Exception as e: # Catch other potential errors
             # print(f"Unexpected error computing reward: {e}. State: {state}. Returning 0.") # Debug if needed
             return 0.0

    def compute_feature_expectations(self, trajectories: List[Dict[str, np.ndarray]]) -> torch.Tensor:
        """Compute discounted feature expectations using NORMALIZED features."""
        total_discounted_sum = torch.zeros(self.n_features, device=self.device)
        n_valid_trajectories = 0
        total_valid_steps = 0

        for traj in trajectories:
            # Use pre-filtered states s0..s(T-1)
            if "states" not in traj or not isinstance(traj["states"], np.ndarray) or len(traj["states"]) == 0:
                continue
            states = traj["states"] # Already validated in _filter_valid_trajectories

            n_valid_trajectories += 1
            discounted_sum = torch.zeros(self.n_features, device=self.device)
            trajectory_valid_steps = 0
            for t, state in enumerate(states):
                try:
                    features = get_state_features(state)
                    normalized_features = self.normalize_features(features)
                    features_tensor = torch.FloatTensor(normalized_features).to(self.device)

                    # Check for NaNs/Infs *after* normalization
                    if torch.isnan(features_tensor).any() or torch.isinf(features_tensor).any():
                         # print(f"Skipping step {t} due to NaN/Inf features") # Reduce verbosity
                         continue # Skip this state

                    discount_factor = self.gamma ** t
                    discounted_sum += discount_factor * features_tensor
                    trajectory_valid_steps += 1
                except ValueError: continue # Skip state if get_state_features fails
                except Exception: continue # Skip state on other errors

            if trajectory_valid_steps > 0:
                total_discounted_sum += discounted_sum
                total_valid_steps += trajectory_valid_steps
            # else: # Trajectory had no valid steps, don't count it
            #     n_valid_trajectories -= 1

        if n_valid_trajectories == 0:
            print("Warning: No valid trajectories contributed to feature expectation calculation.")
            return torch.zeros(self.n_features, device=self.device)

        # Average over trajectories that contributed
        result_mu = total_discounted_sum / n_valid_trajectories
        # Sanity check final result
        if torch.isnan(result_mu).any() or torch.isinf(result_mu).any():
            print("Warning: Final computed feature expectation contains NaN/Inf. Returning zero.")
            return torch.zeros(self.n_features, device=self.device) # Return zero vector on failure

        return result_mu


    def update_reward_parameters(self, current_mu: torch.Tensor) -> float:
        """Update reward weights using projection method (operates on normalized mu)."""
        # Check incoming mu
        if torch.isnan(current_mu).any() or torch.isinf(current_mu).any():
            print("Warning: Received NaN/Inf current_mu. Skipping weight update.")
            return float('inf') # Indicate failure

        current_mu_device = current_mu.detach().clone().to(self.device)
        self.previous_feature_expectations.append(current_mu_device)

        min_dist_sq = float('inf'); mu_bar = None
        expert_mu_device = self.expert_feature_expectations.to(self.device) # Already normalized from init

        # Check expert mu validity *once*
        if torch.isnan(expert_mu_device).any() or torch.isinf(expert_mu_device).any():
             print("Error: Expert MU is invalid. Cannot update weights.")
             return float('inf')

        # Filter history for valid mus
        valid_history_mus = [
            mu.to(self.device) for mu in self.previous_feature_expectations
            if not (torch.isnan(mu).any() or torch.isinf(mu).any())
        ]

        if not valid_history_mus:
             print("Warning: No valid mu in history. Using current_mu as mu_bar (if valid).")
             # Use current_mu if it's valid
             if not (torch.isnan(current_mu_device).any() or torch.isinf(current_mu_device).any()):
                 mu_bar = current_mu_device
                 min_dist_sq = torch.sum((expert_mu_device - mu_bar)**2)
             else:
                 print("Error: Current MU is also invalid. Cannot find mu_bar.")
                 return float('inf') # Cannot proceed
        else:
             try:
                 # Calculate squared distances efficiently
                 history_stack = torch.stack(valid_history_mus)
                 diffs = expert_mu_device.unsqueeze(0) - history_stack
                 dists_sq = torch.sum(diffs**2, dim=1)

                 # Find the minimum distance and corresponding mu_bar
                 min_dist_sq, min_idx = torch.min(dists_sq, dim=0)
                 mu_bar = valid_history_mus[min_idx]

             except RuntimeError as e:
                 print(f"Error during distance calculation: {e}")
                 # Fallback: use current mu if valid
                 if not (torch.isnan(current_mu_device).any() or torch.isinf(current_mu_device).any()):
                     print("Falling back to using current MU as mu_bar.")
                     mu_bar = current_mu_device
                     min_dist_sq = torch.sum((expert_mu_device - mu_bar)**2)
                 else:
                      print("Error: Fallback current MU is also invalid.")
                      return float('inf')
             except Exception as e:
                 print(f"Unexpected error finding mu_bar: {e}")
                 return float('inf')


        # Ensure mu_bar is valid before weight update
        if mu_bar is None or torch.isnan(mu_bar).any() or torch.isinf(mu_bar).any():
             print("Error: Selected mu_bar is invalid or None. Cannot update weights.")
             return float('inf')

        # Update weights: w = mu_E - mu_bar
        self.reward_weights = expert_mu_device - mu_bar

        # Normalize weights
        norm = torch.norm(self.reward_weights)
        if norm > 1e-8:
            self.reward_weights /= norm
        else:
            # If norm is zero (mu_E == mu_bar), set weights to zero explicitly
            self.reward_weights = torch.zeros_like(self.reward_weights)

        # Return the distance (sqrt of min squared distance)
        # Clamp before sqrt to avoid potential issues with tiny negative values due to precision
        return torch.sqrt(torch.clamp(min_dist_sq, min=0.0)).item()


    def sample_action(self, state_tensor_normalized: torch.Tensor, deterministic: bool = False, use_policy: bool = True) -> int:
        """Sample action using policy network, with exploration."""
        # Epsilon-greedy exploration or random action
        if not use_policy or (not deterministic and random.random() < self.exploration_rate):
            return random.randint(0, self.n_actions - 1)
        else:
            # Use policy network
            self.policy.eval() # Ensure policy is in eval mode for sampling
            with torch.no_grad():
                # Check input tensor validity
                if torch.isnan(state_tensor_normalized).any() or torch.isinf(state_tensor_normalized).any():
                    # print("Warning: NaN/Inf input to sample_action. Returning random.") # Reduce verbosity
                    return random.randint(0, self.n_actions - 1)

                logits = self.policy(state_tensor_normalized)

                # Check output logits validity
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    # print(f"Warning: NaN/Inf logits. Returning random.") # Reduce verbosity
                    return random.randint(0, self.n_actions - 1)

                # Sample or take greedy action
                if deterministic:
                    action = torch.argmax(logits, dim=-1).item()
                else:
                    # Use Categorical for sampling from probabilities derived from logits
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()
                return action


    def sample_trajectories(self, n_trajectories: int, deterministic: bool = False, use_current_policy: bool = True, max_steps_override = None) -> List[Dict[str, np.ndarray]]:
        """Sample trajectories using the current policy."""
        trajectories = []
        max_steps = max_steps_override if max_steps_override else (self.env.spec.max_episode_steps if self.env.spec and self.env.spec.max_episode_steps else 1000)

        for i in range(n_trajectories):
            state = None; states = []; actions = []; learned_rewards = []; env_rewards = []
            try:
                state, _ = self.env.reset()
                if state is None or np.isnan(state).any() or np.isinf(state).any():
                    # print(f"Warning: Invalid initial state from env.reset(). Skipping trajectory {i+1}.")
                    continue # Skip if reset fails or returns invalid state
            except Exception as e:
                 # print(f"Warning: Error during env.reset() for trajectory {i+1}: {e}. Skipping.")
                 continue # Skip if reset fails

            done = False
            for step in range(max_steps):
                 # State validity check before using it
                if state is None or np.isnan(state).any() or np.isinf(state).any() or len(state) != self.state_dim :
                    # print(f"Warning: Invalid state encountered mid-trajectory {i+1} at step {step}. Terminating.")
                    done = True; break # Terminate trajectory if state becomes invalid
                states.append(state.copy()) # Store valid state

                # Normalize state for policy (includes NaN/Inf check)
                state_normalized = self.normalize_state(state)

                # Check normalized state before converting to tensor
                if np.isnan(state_normalized).any() or np.isinf(state_normalized).any():
                    # print(f"Warning: Invalid normalized state in trajectory {i+1} at step {step}. Terminating.")
                    done = True; break # Terminate if normalization fails

                state_tensor = torch.FloatTensor(state_normalized).to(self.device).unsqueeze(0)

                # Sample action (includes NaN/Inf check)
                action = self.sample_action(state_tensor, deterministic=deterministic, use_policy=use_current_policy)

                # Compute learned reward (includes NaN/Inf check)
                irl_reward = self.compute_reward(state)
                learned_rewards.append(irl_reward)

                # Step environment
                next_state = None; env_reward = 0.0; terminated = False; truncated = False
                try:
                    next_state, env_reward, terminated, truncated, _ = self.env.step(action)
                    # Basic validity check on next_state
                    if next_state is None:
                        # print(f"Warning: env.step returned None next_state in trajectory {i+1}. Terminating.")
                        done = True
                        next_state = state # Use last valid state to avoid errors below
                    elif np.isnan(next_state).any() or np.isinf(next_state).any():
                        # print(f"Warning: env.step returned NaN/Inf next_state in trajectory {i+1}. Terminating.")
                        done = True
                        next_state = state # Use last valid state
                except Exception as e: # Catch errors during step
                    # print(f"Warning: Error during env.step() in trajectory {i+1}: {e}. Terminating.")
                    done = True; next_state = state # Use last valid state if step fails

                done = terminated or truncated # Update done flag based on step result
                actions.append(action)
                env_rewards.append(env_reward)
                state = next_state # Update state for the next iteration

                if done: break # Exit inner loop if terminated/truncated/error

            # Store trajectory if valid steps were taken
            if states and actions:
                 # Ensure lists have consistent length (should be equal if loop completes normally)
                 min_len = min(len(states), len(actions), len(learned_rewards), len(env_rewards))
                 # Only include trajectories with at least one step
                 if min_len > 0:
                     trajectories.append({
                         "states": np.array(states[:min_len], dtype=np.float32),
                         "actions": np.array(actions[:min_len], dtype=np.int64),
                         "learned_rewards": np.array(learned_rewards[:min_len], dtype=np.float32),
                         "env_rewards": np.array(env_rewards[:min_len], dtype=np.float32)
                     })
        return trajectories


    def optimize_policy(self, trajectories: List[Dict[str, np.ndarray]]) -> float:
        """Optimize policy using Policy Gradient with G_t based on learned rewards."""
        total_loss = 0.0; n_updates = 0
        all_norm_states, all_actions, all_discounted_returns = [], [], []

        # --- Data Collection and Preparation ---
        for traj in trajectories:
            # Validate trajectory data needed for optimization
            if not all(k in traj for k in ["states", "actions", "learned_rewards"]) or \
               len(traj["states"]) == 0 or \
               len(traj["states"]) != len(traj["actions"]) or \
               len(traj["states"]) != len(traj["learned_rewards"]):
                # print("Warning: Skipping malformed trajectory in optimize_policy.") # Reduce verbosity
                continue # Skip malformed trajectories

            states = traj["states"]; actions = traj["actions"]; learned_rewards = traj["learned_rewards"]

            # Check for invalid rewards before calculating returns
            if np.isnan(learned_rewards).any() or np.isinf(learned_rewards).any():
                # print("Warning: NaN/Inf learned_rewards found. Skipping trajectory.") # Reduce verbosity
                continue

            # Calculate discounted returns G_t = sum_{k=t}^T gamma^{k-t} * r_k
            returns = np.zeros_like(learned_rewards, dtype=np.float32); discounted_return = 0.0
            for t in reversed(range(len(learned_rewards))):
                discounted_return = learned_rewards[t] + self.gamma * discounted_return
                returns[t] = discounted_return

            # Normalize returns (acts as advantage estimate) per trajectory
            if len(returns) > 1:
                 mean_ret, std_ret = np.mean(returns), np.std(returns)
                 # Avoid division by zero if std is very small
                 returns = (returns - mean_ret) / (std_ret + 1e-8)
            elif len(returns) == 1:
                 returns = returns - np.mean(returns) # Just center if single step

             # Check for invalid returns after normalization
            if np.isnan(returns).any() or np.isinf(returns).any():
                 # print("Warning: NaN/Inf returns after normalization. Skipping trajectory.") # Reduce verbosity
                 continue

            # Normalize states for policy input (includes NaN/Inf check)
            norm_states = self.normalize_state(states)
            if np.isnan(norm_states).any() or np.isinf(norm_states).any():
                # print("Warning: NaN/Inf normalized states found. Skipping trajectory.") # Reduce verbosity
                continue

            # Append valid data to lists
            all_norm_states.extend(norm_states); all_actions.extend(actions); all_discounted_returns.extend(returns)

        # Check if any valid data was collected
        if not all_norm_states:
            print("Warning: No valid data collected for policy optimization.")
            return 0.0 # No data to train on

        # --- Batch Training Loop ---
        try:
            # Convert collected data to tensors
            states_tensor = torch.FloatTensor(np.array(all_norm_states)).to(self.device)
            actions_tensor = torch.LongTensor(np.array(all_actions)).to(self.device)
            returns_tensor = torch.FloatTensor(np.array(all_discounted_returns)).to(self.device)
        except Exception as e:
            print(f"Error converting data to tensors in optimize_policy: {e}")
            return 0.0 # Avoid crashing

        n_samples = len(states_tensor)
        if n_samples == 0: return 0.0
        indices = np.arange(n_samples)

        self.policy.train() # Set policy to training mode
        for epoch in range(self.n_policy_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, n_samples, self.policy_batch_size):
                end_idx = min(start_idx + self.policy_batch_size, n_samples)
                batch_indices = indices[start_idx : end_idx]
                if len(batch_indices) == 0: continue

                # Get batch data
                state_batch = states_tensor[batch_indices]
                action_batch = actions_tensor[batch_indices]
                return_batch = returns_tensor[batch_indices] # G_t (normalized)

                # Policy forward pass
                logits = self.policy(state_batch)
                dist = torch.distributions.Categorical(logits=logits)

                # Calculate loss components
                log_probs = dist.log_prob(action_batch)
                entropy = dist.entropy().mean()
                policy_gradient_loss = -torch.mean(log_probs * return_batch) # REINFORCE objective
                entropy_loss = - 0.01 * entropy # Entropy bonus to encourage exploration

                loss = policy_gradient_loss + entropy_loss

                # Check for invalid loss before backward pass
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                     # print(f"Warning: NaN/Inf loss detected in epoch {epoch+1}. Skipping batch.") # Reduce verbosity
                     continue

                # Backward pass and optimizer step
                self.policy_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0) # Gradient clipping
                self.policy_optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        self.policy.eval() # Set policy back to evaluation mode after training
        avg_loss = total_loss / n_updates if n_updates > 0 else 0.0
        return avg_loss


    def train(self, n_iterations: int, n_trajectories_per_iter: int,
              reward_threshold: float = 200, eval_episodes: int = 10, eval_interval: int = 10) -> Dict:
        """Main training loop for MaxMargin IRL."""
        stats = {
            'eval_rewards_mean': [], 'eval_rewards_std': [],
            'irl_distances': [], 'policy_losses': [],
            'exploration_rates': [], 'weights_norm': [],
            'weights_history': [] # Store actual weight vectors over time
        }
        best_eval_reward = -float('inf')
        self.eval_interval = eval_interval # Store for plotting reference

        print(f"\nStarting MaxMargin IRL training for {n_iterations} iterations...")

        # Initial evaluation before training starts
        initial_eval_reward, initial_eval_std = self.evaluate_policy(eval_episodes)
        print(f"Initial Avg Eval Reward: {initial_eval_reward:.2f} +/- {initial_eval_std:.2f}")
        # Record initial state for iteration 0
        stats['eval_rewards_mean'].append(initial_eval_reward)
        stats['eval_rewards_std'].append(initial_eval_std)
        stats['irl_distances'].append(0) # No distance before first update
        stats['policy_losses'].append(0) # No loss before first optimization
        stats['exploration_rates'].append(self.exploration_rate)
        stats['weights_norm'].append(torch.norm(self.reward_weights).item())
        stats['weights_history'].append(self.reward_weights.detach().cpu().numpy().copy())


        for i in range(n_iterations):
            iter_start_time = time.time(); print(f"\n--- Iteration {i+1}/{n_iterations} ---")

            # --- Step 1: Sample Trajectories using Current Policy (with Exploration) ---
            print(f"Sampling {n_trajectories_per_iter} trajectories (explore rate: {self.exploration_rate:.3f})...");
            # Ensure policy is in eval mode for sampling, but allow epsilon-greedy
            self.policy.eval()
            sampled_trajectories = self.sample_trajectories(n_trajectories_per_iter, use_current_policy=True, deterministic=False)

            if not sampled_trajectories:
                print("Warning: Failed to sample any valid trajectories. Skipping iteration.")
                # Append Nones or last values to stats to keep length consistent? Or skip stat update?
                # Let's append last values where appropriate, indicate failure elsewhere.
                if stats['policy_losses']: stats['policy_losses'].append(stats['policy_losses'][-1])
                else: stats['policy_losses'].append(None)
                if stats['irl_distances']: stats['irl_distances'].append(stats['irl_distances'][-1])
                else: stats['irl_distances'].append(None)
                stats['exploration_rates'].append(self.exploration_rate) # Exploration still happened (attempted)
                if stats['weights_norm']: stats['weights_norm'].append(stats['weights_norm'][-1])
                else: stats['weights_norm'].append(None)
                if stats['weights_history']: stats['weights_history'].append(stats['weights_history'][-1]) # Weights didn't change
                else: stats['weights_history'].append(None)
                # Don't evaluate if sampling failed
                if stats['eval_rewards_mean']: stats['eval_rewards_mean'].append(stats['eval_rewards_mean'][-1]); stats['eval_rewards_std'].append(stats['eval_rewards_std'][-1])

                continue # Move to next iteration


            # --- Step 2: Optimize Policy using Learned Rewards from Sampled Trajectories ---
            # Note: The rewards used here are based on the weights *before* the update in this iteration.
            # This is a common simplification in MaxMargin implementations.
            print(f"Optimizing policy ({self.n_policy_epochs} epochs)...");
            # optimize_policy already handles reward calculation internally based on current weights
            policy_loss = self.optimize_policy(sampled_trajectories);
            stats['policy_losses'].append(policy_loss)


            # --- Step 3: Compute Feature Expectations of the *Sampled* Trajectories ---
            # These expectations reflect the behavior of the policy used for sampling (pi_i)
            print("Computing feature expectations (normalized)...");
            current_mu = self.compute_feature_expectations(sampled_trajectories)

            # Check mu validity before proceeding
            if torch.isnan(current_mu).any() or torch.isinf(current_mu).any():
                print("Warning: NaN/Inf MU computed after policy optimization. Skipping weight update for this iteration.")
                # Record failure/last value for stats
                stats['irl_distances'].append(stats['irl_distances'][-1] if stats['irl_distances'] else None)
                stats['weights_norm'].append(stats['weights_norm'][-1] if stats['weights_norm'] else None)
                stats['weights_history'].append(stats['weights_history'][-1] if stats['weights_history'] else None)
                # Don't evaluate if MU is invalid
                if stats['eval_rewards_mean']: stats['eval_rewards_mean'].append(stats['eval_rewards_mean'][-1]); stats['eval_rewards_std'].append(stats['eval_rewards_std'][-1])
                stats['exploration_rates'].append(self.exploration_rate) # Record exploration rate
                continue # Skip to next iteration

            # --- Step 4: Update Reward Weights using Projection ---
            # w_{i+1} = mu_E - mu_bar, where mu_bar is closest in history to mu_E
            print("Updating reward weights...");
            irl_distance = self.update_reward_parameters(current_mu); # Handles mu validity check internally
            stats['irl_distances'].append(irl_distance)
            current_weights_norm = torch.norm(self.reward_weights).item();
            stats['weights_norm'].append(current_weights_norm);
            stats['weights_history'].append(self.reward_weights.detach().cpu().numpy().copy())
            print(f"IRL Dist: {irl_distance:.4f}, Weights Norm: {current_weights_norm:.3f}")


            # --- Step 5: Evaluate the *Updated* Policy Periodically ---
            if (i + 1) % eval_interval == 0 or i == n_iterations - 1:
                print("Evaluating policy...");
                # Use the policy network that was just optimized
                avg_reward, std_reward = self.evaluate_policy(eval_episodes)
                stats['eval_rewards_mean'].append(avg_reward); stats['eval_rewards_std'].append(std_reward)
                print(f"Iter {i+1}: Eval Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
                if avg_reward > best_eval_reward:
                    best_eval_reward = avg_reward; print(f"New best average eval reward: {best_eval_reward:.2f}")
                    # Optional: Save the best performing policy state dict
                    # try: torch.save(self.policy.state_dict(), "irl_lander_best_policy.pth"); print("Saved best policy.")
                    # except Exception as e: print(f"Error saving best policy: {e}")

                # Check for termination condition
                if best_eval_reward >= reward_threshold:
                    print(f"Reward threshold ({reward_threshold}) reached or exceeded. Stopping training.")
                    # Record exploration rate for this final iteration before breaking
                    stats['exploration_rates'].append(self.exploration_rate)
                    break # Exit training loop
            else:
                 # If not evaluating, append the last recorded eval reward to keep plot alignment
                 if stats['eval_rewards_mean']: # Check if there's a previous value
                     stats['eval_rewards_mean'].append(stats['eval_rewards_mean'][-1])
                     stats['eval_rewards_std'].append(stats['eval_rewards_std'][-1])
                 # Else: this should only happen if eval_interval > 1 and it's the first few iters


            # --- Step 6: Anneal Exploration Rate ---
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay);
            stats['exploration_rates'].append(self.exploration_rate) # Record after potential update

            iter_time = time.time() - iter_start_time; print(f"Iter {i+1} time: {iter_time:.2f}s.")

        # --- Training Loop Finished ---

        # Perform a final evaluation if the loop finished naturally (not stopped early by threshold)
        if not (best_eval_reward >= reward_threshold and i < n_iterations -1) :
             print("\nPerforming final evaluation after training loop completed...")
             avg_reward, std_reward = self.evaluate_policy(eval_episodes * 2) # More eps for final eval
             print(f"Final Avg Eval Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
             if avg_reward > best_eval_reward: best_eval_reward = avg_reward

        # Ensure all stats lists have the same length (pad if loop broke early)
        # This usually isn't strictly necessary if plotting handles different lengths, but good practice.
        max_len = n_iterations + 1 # +1 for initial values
        for key in stats:
            current_len = len(stats[key])
            if current_len < max_len:
                padding_needed = max_len - current_len
                last_val = stats[key][-1] if current_len > 0 else None
                stats[key].extend([last_val] * padding_needed)


        return {
            "policy": self.policy,
            "reward_weights": self.reward_weights.detach().cpu().numpy(),
            "stats": stats,
            "best_eval_reward": best_eval_reward,
            "state_mean": self.state_mean, "state_std": self.state_std,
            "feature_mean": self.feature_mean, "feature_std": self.feature_std
        }


    def evaluate_policy(self, n_episodes):
        """Evaluate the current IRL policy deterministically."""
        self.policy.eval(); all_rewards = []
        eval_env = None # Initialize outside try block
        try:
            # Use the same env ID as the training env
            env_id = self.env.spec.id if self.env.spec else 'LunarLander-v3'
            eval_env = gym.make(env_id)
        except Exception as e:
            print(f"Eval env creation failed ({env_id}): {e}")
            return -float('inf'), 0.0 # Cannot evaluate

        for episode in range(n_episodes):
            state = None; ep_reward = 0; done = False
            try:
                state, _ = eval_env.reset()
                if state is None or np.isnan(state).any() or np.isinf(state).any():
                    # print(f"Warning: Invalid initial state in eval episode {episode+1}. Skipping.")
                    continue # Skip episode if reset fails
            except Exception as e:
                 # print(f"Warning: Error during eval env.reset() for episode {episode+1}: {e}. Skipping.")
                 continue

            max_steps = eval_env.spec.max_episode_steps if eval_env.spec and eval_env.spec.max_episode_steps else 1000
            for _step in range(max_steps):
                 # Check state validity before use
                 if state is None or np.isnan(state).any() or np.isinf(state).any() or len(state) != self.state_dim:
                      # print(f"Warning: Invalid state during eval episode {episode+1}, step {_step}. Terminating.")
                      break # End episode if state becomes invalid
                 state_normalized = self.normalize_state(state)

                 # Check normalized state validity
                 if np.isnan(state_normalized).any() or np.isinf(state_normalized).any():
                      # print(f"Warning: Invalid normalized state during eval episode {episode+1}, step {_step}. Terminating.")
                      break

                 state_tensor = torch.FloatTensor(state_normalized).to(self.device).unsqueeze(0)
                 action = self.sample_action(state_tensor, deterministic=True, use_policy=True) # Deterministic eval

                 try:
                     next_state, reward, terminated, truncated, _ = eval_env.step(action)
                     # Check next state validity
                     if next_state is None or np.isnan(next_state).any() or np.isinf(next_state).any():
                         # print(f"Warning: Invalid next_state from step in eval episode {episode+1}. Terminating.")
                         done = True; next_state = state # Use last valid state
                     else:
                         done = terminated or truncated
                 except Exception as e:
                      # print(f"Warning: Error during eval env.step() in episode {episode+1}: {e}. Terminating.")
                      done = True; next_state = state # Handle step error, use last valid state

                 ep_reward += reward; state = next_state
                 if done: break
            all_rewards.append(ep_reward)

        if eval_env: eval_env.close() # Close env if successfully created

        mean_reward = np.mean(all_rewards) if all_rewards else -float('inf') # Return -inf if no episodes completed
        std_reward = np.std(all_rewards) if all_rewards else 0.0
        return mean_reward, std_reward


# --- PPO Expert Code (Adapted from reference) ---
class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim), # Output logits
        )
    def forward(self, x): return self.network(x)

class PPOCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1) # Output single value
        )
    def forward(self, x): return self.network(x)

# --- PPO Expert Code (Adapted from reference) ---
class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim), # Output logits
        )
    def forward(self, x): return self.network(x)

class PPOCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1) # Output single value
        )
    def forward(self, x): return self.network(x)

class PPOExpert:
    """ PPO Expert Trainer and Policy """
    def __init__(self, env, device="cpu", model_path='ppo_expert_lander.pth'):
        self.env = env
        self.env_id = env.spec.id if env.spec else 'LunarLander-v3' # Store env ID
        self.device = device
        self.model_path = model_path
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.actor = PPOActor(self.state_dim, self.action_dim).to(device)
        self.critic = PPOCritic(self.state_dim).to(device)
        # Consider slightly lower learning rates if unstable
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # PPO Hyperparameters (typical values)
        self.n_steps = 2048        # Steps per environment interaction before update (Reduced)
        self.n_epochs = 10         # Optimization epochs per update (Reduced)
        self.batch_size = 64       # Minibatch size for optimization (Reduced to match common SB3 defaults)
        self.gamma = 0.99          # Discount factor
        self.gae_lambda = 0.95     # Factor for Generalized Advantage Estimation
        self.clip_epsilon = 0.2    # PPO clipping parameter
        self.ent_coef = 0.01       # Entropy coefficient for exploration bonus
        self.vf_coef = 0.5         # Value function loss coefficient (Can be used if desired)
        self.max_grad_norm = 0.5   # Max gradient norm for clipping

        # Training Control
        # Reduced timesteps as convergence should be faster if working
        self.n_timesteps = 1_000_000
        self.target_reward = 270   # Reward threshold to stop training early
        self.max_std_dev = 20      # Maximum allowed standard deviation for early stopping
        self.eval_episodes = 20    # Episodes for evaluation
        self.log_interval = 10     # How many updates between logs/evals
        self.best_score = -float('inf') # Track best evaluation score

    def select_action(self, state, deterministic=False):
        """ Select action using the actor network """
        # No need to set eval mode here if called from train/evaluate which manage it
        with torch.no_grad():
            state_np = np.asarray(state, dtype=np.float32)
            # Add batch dimension if missing
            if state_np.ndim == 1: state_np = state_np[np.newaxis, :]
            state_tensor = torch.FloatTensor(state_np).to(self.device)

            # Check for NaNs in input state
            if torch.isnan(state_tensor).any():
                print("Warning: NaN state input to select_action. Returning random.")
                # Return a default action, maybe 0, or random
                action_dim = self.actor.network[-1].out_features
                action = torch.randint(0, action_dim, (state_tensor.shape[0],), device=self.device)
                # Need dummy log_prob and value - might cause issues downstream if not handled
                log_prob = torch.zeros(state_tensor.shape[0], device=self.device)
                value = torch.zeros(state_tensor.shape[0], 1, device=self.device)
            else:
                logits = self.actor(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)

                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    action = dist.sample()

                log_prob = dist.log_prob(action)
                value = self.critic(state_tensor) # Critic value for this state

            # Squeeze results if input was single state
            if state_np.shape[0] == 1:
                return action.item(), log_prob.squeeze().detach(), value.squeeze().detach() # DETACH HERE
            else: # Keep batch dimension if input was batched
                return action, log_prob.detach(), value.detach() # DETACH HERE

    def compute_gae(self, rewards, values, next_value, dones):
        """ Compute Generalized Advantage Estimation """
        advantages = torch.zeros_like(rewards).to(self.device)
        gae = 0.0
        # Ensure next_value is a tensor and has the correct shape/device
        # next_value is already detached from select_action
        if not isinstance(next_value, torch.Tensor):
             next_value = torch.tensor(next_value, dtype=torch.float32).to(self.device)
        next_value = next_value.reshape(-1, 1) # Ensure it's [1, 1] or similar

        # Combine values and next_value for easier indexing
        # values should already be [n_steps, 1] and detached
        extended_values = torch.cat([values, next_value], dim=0) # Shape [n_steps+1, 1]

        for t in reversed(range(len(rewards))):
            # Ensure dones[t] is float tensor
            done_mask = 1.0 - dones[t] # Assuming dones are already float tensors on device
            # Calculate TD error (delta)
            delta = rewards[t] + self.gamma * extended_values[t+1] * done_mask - extended_values[t]
            # Update GAE
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae
            advantages[t] = gae

        # Calculate returns: Q(s,a) estimate = A(s,a) + V(s)
        # values here is the [n_steps, 1] tensor from buffer
        returns = advantages.unsqueeze(1) + values # Ensure returns is [n_steps, 1]

        # Normalize advantages (often done for stability) - across the whole rollout
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns # advantages=[n_steps], returns=[n_steps, 1]

    def evaluate(self):
        """ Evaluate the current PPO policy deterministically """
        self.actor.eval(); self.critic.eval(); rewards = []
        eval_env = None
        try: eval_env = gym.make(self.env_id) # Use stored env ID
        except Exception as e: print(f"PPO Eval env creation failed: {e}"); return -float('inf'), 0.0

        for _ in range(self.eval_episodes):
            state, _ = eval_env.reset()
            if state is None or np.isnan(state).any() or np.isinf(state).any(): continue # Skip if reset fails
            ep_reward = 0; done = False
            max_steps = eval_env.spec.max_episode_steps if eval_env.spec else 1000
            for _step in range(max_steps):
                 # Use internal select_action which handles device/batching/NaNs/detaching
                action, _, _ = self.select_action(state, deterministic=True)
                try:
                    next_state, reward, terminated, truncated, _ = eval_env.step(action)
                    if next_state is None or np.isnan(next_state).any() or np.isinf(next_state).any():
                         # print(f"Warning: PPO Eval step returned invalid state. Terminating ep.") # Reduce verbosity
                         done = True; next_state = state # Use last valid state
                    else: done = terminated or truncated
                except Exception as e:
                    print(f"Warning: PPO Eval step error: {e}. Terminating ep.")
                    done = True; next_state = state # End episode on error
                ep_reward += reward; state = next_state
                if done: break
            rewards.append(ep_reward)

        if eval_env: eval_env.close()
        # Set back to train mode implicitly handled by train() call if resuming
        # self.actor.train(); self.critic.train(); # Set back to train mode
        mean_rew = np.mean(rewards) if rewards else -float('inf')
        std_rew = np.std(rewards) if rewards else 0.0
        return mean_rew, std_rew

    def train(self):
        """ Main PPO training loop """
        print(f"Training PPO expert for ~{self.n_timesteps} timesteps on {self.env_id}...");
        print(f"Hyperparams: n_steps={self.n_steps}, n_epochs={self.n_epochs}, batch_size={self.batch_size}")
        total_timesteps = 0; update_count = 0

        # Initialize buffers
        buffers = {
            'states': torch.zeros((self.n_steps, self.state_dim), dtype=torch.float32).to(self.device),
            'actions': torch.zeros((self.n_steps,), dtype=torch.int64).to(self.device),
            'log_probs': torch.zeros((self.n_steps,), dtype=torch.float32).to(self.device), # From OLD policy
            'rewards': torch.zeros((self.n_steps,), dtype=torch.float32).to(self.device),
            'dones': torch.zeros((self.n_steps,), dtype=torch.float32).to(self.device),
            'values': torch.zeros((self.n_steps, 1), dtype=torch.float32).to(self.device) # From OLD policy critic
        }

        try: state, _ = self.env.reset()
        except Exception as e: print(f"Initial PPO env reset failed: {e}"); return # Cannot start training
        if state is None or np.isnan(state).any() or np.isinf(state).any(): print("Initial PPO state invalid."); return

        while total_timesteps < self.n_timesteps:
            # Ensure train mode for rollout generation using current policy
            self.actor.train(); self.critic.train()

            # --- Rollout Phase ---
            for step in range(self.n_steps):
                # Ensure state is valid numpy array before passing to select_action
                if state is None or not isinstance(state, np.ndarray) or np.isnan(state).any() or np.isinf(state).any():
                    print("Warning: Invalid state detected before action selection. Resetting environment.")
                    try: state, _ = self.env.reset(); reward = -100.0; done = True # Penalize heavily
                    except Exception as e: print(f"PPO env reset failed after invalid state: {e}"); return
                    if state is None or np.isnan(state).any() or np.isinf(state).any(): print("PPO state invalid after reset."); return
                    # Need to decide how to handle this step - skip? store dummy data?
                    # For simplicity, let's store dummy data for this step and reset state below
                    action, log_prob, value = 0, torch.tensor(0.0), torch.tensor([[0.0]]) # Dummy values
                else:
                    # Select action, get log_prob and value (log_prob, value are detached inside)
                    action, log_prob, value = self.select_action(state, deterministic=False) # Returns detached log_prob, value

                # Execute action in environment
                try:
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    # Check for invalid next_state
                    if next_state is None or np.isnan(next_state).any() or np.isinf(next_state).any():
                        print("Warning: Invalid next_state during PPO rollout. Resetting.")
                        try: next_state, _ = self.env.reset(); reward = -100.0; done = True # Penalize and reset
                        except Exception as reset_e: print(f"PPO env reset failed after invalid next_state: {reset_e}"); return
                    elif done: # Handle normal termination/truncation
                         try: next_state, _ = self.env.reset()
                         except Exception as reset_e: print(f"PPO env reset failed after done: {reset_e}"); return
                except Exception as e:
                    print(f"Warning: PPO Train step error: {e}. Resetting.")
                    try: next_state, _ = self.env.reset(); reward = -100.0; done = True # Penalize heavily on error
                    except Exception as reset_e: print(f"PPO env reset failed after step error: {reset_e}"); return

                # Store transition in buffer (use state observed *before* action)
                # Ensure state was valid before storing
                state_tensor = torch.FloatTensor(np.asarray(state)).to(self.device)
                if torch.isnan(state_tensor).any(): # Check again in case invalid state wasn't caught above
                    print("Warning: Storing NaN state in buffer. Replacing with zeros.")
                    state_tensor = torch.zeros_like(state_tensor)

                buffers['states'][step] = state_tensor
                buffers['actions'][step] = torch.tensor(action, dtype=torch.int64).to(self.device) # Action can be tensor or int
                buffers['log_probs'][step] = log_prob # Already detached
                buffers['rewards'][step] = torch.tensor(reward, dtype=torch.float32).to(self.device)
                buffers['dones'][step] = torch.tensor(float(done), dtype=torch.float32).to(self.device)
                buffers['values'][step] = value # Already detached, shape [1] or [1,1], ensure [n_steps, 1] final shape

                state = next_state # Move to next state
                total_timesteps += 1


            # --- Compute Advantages and Returns ---
            with torch.no_grad():
                # Get value of the final next_state (already reset if done)
                _, _, next_value = self.select_action(state, deterministic=False) # Detached value

            # Ensure values buffer has the correct shape [n_steps, 1] before passing
            # values_buffer shape is already (n_steps, 1) from init
            advantages, returns = self.compute_gae(buffers['rewards'], buffers['values'], next_value, buffers['dones'])
            # advantages shape [n_steps], returns shape [n_steps, 1]

            # Flatten buffers for batching (advantages/returns already correct size)
            b_states = buffers['states'].view(-1, self.state_dim)
            b_actions = buffers['actions'].view(-1)
            b_log_probs = buffers['log_probs'].view(-1) # Old log probs
            b_values = buffers['values'].view(-1)       # Old values (for reference if needed, not directly used in loss)
            b_advantages = advantages.view(-1)         # Shape [n_steps]
            b_returns = returns.view(-1)               # Shape [n_steps], target for critic

            # --- Optimization Phase ---
            indices = np.arange(self.n_steps)
            actor_losses, critic_losses, entropy_bonuses = [], [], [] # Track losses per update

            for _ in range(self.n_epochs):
                np.random.shuffle(indices)
                for start in range(0, self.n_steps, self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    if len(batch_indices) == 0: continue

                    # Get minibatch data
                    mb_states = b_states[batch_indices]
                    mb_actions = b_actions[batch_indices]
                    mb_old_log_probs = b_log_probs[batch_indices]
                    mb_advantages = b_advantages[batch_indices]
                    mb_returns = b_returns[batch_indices] # Target for value function

                    # Evaluate current policy on batch states (Actor and Critic)
                    # Important: Recompute values and log_probs with current policy params
                    logits = self.actor(mb_states); dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(mb_actions); entropy = dist.entropy().mean()
                    value_preds = self.critic(mb_states).view(-1) # Ensure value pred is flat [batch_size]

                    # Calculate PPO Actor Loss (Clipped Surrogate Objective)
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -self.ent_coef * entropy # Loss is negative bonus
                    total_actor_loss = actor_loss + entropy_loss # Add entropy loss

                    # Actor Optimizer Step
                    self.actor_optimizer.zero_grad(); total_actor_loss.backward(); torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm); self.actor_optimizer.step()

                    # Calculate PPO Critic Loss (Mean Squared Error vs Returns)
                    # You could also use clipped value loss here if desired, but MSE is common
                    critic_loss = nn.functional.mse_loss(mb_returns, value_preds)

                    # Critic Optimizer Step
                    # Apply vf_coef here if desired
                    self.critic_optimizer.zero_grad(); (self.vf_coef * critic_loss).backward(); torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm); self.critic_optimizer.step()

                    # Store losses for logging
                    actor_losses.append(actor_loss.item()); critic_losses.append(critic_loss.item()); entropy_bonuses.append(-entropy_loss.item())


            update_count += 1

            # --- Logging and Evaluation ---
            if update_count % self.log_interval == 0:
                 avg_actor_loss = np.mean(actor_losses)
                 avg_critic_loss = np.mean(critic_losses)
                 avg_entropy = np.mean(entropy_bonuses) / self.ent_coef if self.ent_coef > 0 else 0

                 avg_reward, std_reward = self.evaluate() # Evaluation uses self.actor.eval() internally
                 print(f"T: {total_timesteps}/{self.n_timesteps}, Update: {update_count}, Eval Reward: {avg_reward:.1f} +/- {std_reward:.1f}, Best: {self.best_score:.1f}")
                 print(f"  Losses: Actor={avg_actor_loss:.3f}, Critic={avg_critic_loss:.3f}, Entropy={avg_entropy:.3f}")

                 if avg_reward > self.best_score:
                     print(f"New best score! Saving model to {self.model_path}")
                     self.best_score = avg_reward
                     self.save_model(self.model_path)
                 # Check termination condition - both reward threshold and standard deviation
                 if avg_reward >= self.target_reward and std_reward <= self.max_std_dev:
                     print(f"Target reward ({self.target_reward}) with low std dev ({std_reward:.1f} <= {self.max_std_dev}) reached! Stopping PPO training.")
                     # Ensure model is saved if it reached target on the last eval
                     if avg_reward > self.best_score: self.save_model(self.model_path)
                     return # Stop training

        print("PPO training finished after reaching max timesteps.");
        # Save final model if it wasn't saved already
        final_eval_score, _ = self.evaluate()
        if final_eval_score > self.best_score:
             print("Saving final model as it's better than the previous best.")
             self.save_model(self.model_path)
        elif not os.path.exists(self.model_path):
             print("Saving final model as no model was saved during training.")
             self.save_model(self.model_path)


    def save_model(self, path):
        """ Save actor and critic state dicts """
        # Ensure directory exists only if path includes one
        dir_name = os.path.dirname(path)
        if dir_name:
             os.makedirs(dir_name, exist_ok=True)
        try:
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'best_score': self.best_score
            }, path)
            # print(f"Saved PPO model to {path}") # Reduce verbosity
        except Exception as e: print(f"Error saving PPO model: {e}")

    def load_model(self, path):
        """ Load actor and critic state dicts """
        if not os.path.exists(path):
            print(f"PPO model not found: {path}")
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.best_score = checkpoint.get('best_score', -float('inf'))
            self.actor.to(self.device) # Ensure model is on correct device
            self.critic.to(self.device)
            self.actor.eval() # Set to eval mode after loading
            self.critic.eval()
            print(f"Loaded PPO expert model from {path}, best recorded score: {self.best_score:.2f}")
            return True
        except KeyError as e: print(f"Error loading PPO model from {path}: Missing key {e}"); return False
        except Exception as e: print(f"Error loading PPO model from {path}: {e}"); return False


# --- Helper for PPO Expert Instance ---
_ppo_expert_instance = None
def get_ppo_expert(env, device, force_retrain=False, model_path='ppo_expert_lander.pth'):
    """ Gets or trains the PPO expert instance """
    global _ppo_expert_instance
    if _ppo_expert_instance is None:
        print("Initializing PPO expert instance...")
        _ppo_expert_instance = PPOExpert(env, device=device, model_path=model_path)
        loaded = False
        if os.path.exists(model_path) and not force_retrain:
            print(f"Attempting to load pre-trained PPO model from {model_path}...")
            loaded = _ppo_expert_instance.load_model(model_path)
            if not loaded: print("Failed to load PPO model, will train from scratch.")
        if not loaded or force_retrain:
            if force_retrain: print("Forcing retraining of PPO model...")
            else: print("No usable PPO model found or load failed. Training new model...")
            _ppo_expert_instance.train() # Train until target or n_timesteps
    return _ppo_expert_instance

def expert_policy_ppo(state, deterministic=True, ppo_expert=None):
    """ Selects action using the PPO expert """
    if ppo_expert is None: raise ValueError("PPO Expert instance needed")
    action, _, _ = ppo_expert.select_action(state, deterministic=deterministic)
    return action

# --- Demonstration Generation (using PPO expert, with filtering) ---
def generate_expert_demos(expert_policy_func, env, n_demos, max_steps=1000, min_demo_reward=None, ppo_expert=None):
    """Generates expert demonstrations using the provided policy function, optionally filtering by reward."""
    if ppo_expert is None: raise ValueError("PPO Expert instance required for demo generation")

    print(f"\nGenerating up to {n_demos} expert demonstrations using PPO expert...")
    if min_demo_reward is not None: print(f"Filtering demos: Only keeping episodes with reward >= {min_demo_reward}")

    expert_demos = []; attempts = 0; kept_rewards = []
    demo_env = None
    try:
        # Use the same env ID as the expert was trained on
        env_id = ppo_expert.env_id
        demo_env = gym.make(env_id)
    except Exception as e: print(f"Error creating demo env ({env_id}): {e}"); return [], -float('inf')

    while len(expert_demos) < n_demos and attempts < n_demos * 5: # Limit attempts
        attempts += 1;
        state, _ = demo_env.reset();
        if state is None or np.isnan(state).any() or np.isinf(state).any(): continue # Skip if reset fails

        states_list = [state.copy()]; actions_list = []; ep_reward = 0; done = False
        for step in range(max_steps):
            action = expert_policy_func(state, deterministic=True, ppo_expert=ppo_expert) # Use deterministic actions
            try:
                next_state, reward, terminated, truncated, _ = demo_env.step(action)
                if next_state is None or np.isnan(next_state).any() or np.isinf(next_state).any():
                    done = True; next_state = state # End on invalid state
                else: done = terminated or truncated
            except Exception:
                done = True; next_state = state # End on step error
            actions_list.append(action); ep_reward += reward; state = next_state.copy(); states_list.append(state);
            if done: break

        # Filter based on reward *before* adding to list
        if min_demo_reward is None or ep_reward >= min_demo_reward:
            # Store raw observations (s0..sT) and actions (a0..aT-1)
            expert_demos.append({
                "observations": np.array(states_list, dtype=np.float32), # Includes final state
                "actions": np.array(actions_list, dtype=np.int64)
            });
            kept_rewards.append(ep_reward)
            # if len(expert_demos) % 20 == 0: print(f"Kept {len(expert_demos)}/{attempts} demos...") # Reduce verbosity

    if demo_env: demo_env.close()
    avg_kept_reward = np.mean(kept_rewards) if kept_rewards else -float('inf');
    print(f"Generated and kept {len(expert_demos)} demos (avg reward: {avg_kept_reward:.2f}) after {attempts} attempts.");
    return expert_demos, avg_kept_reward

# --- Process Demonstrations (Robust version, consistent with IRL class filter) ---
def process_expert_demonstrations(demos: List[Dict]) -> List[Dict[str, np.ndarray]]:
    """
    Process expert demonstrations (expecting 'observations' and 'actions')
    into the format needed by IRL (extracting 'states' s0..s(T-1)).
    Matches filtering logic in MaxMarginIRL._filter_valid_trajectories.
    """
    processed = []; expected_dim = None
    if demos:
        try: expected_dim = demos[0]["observations"].shape[1]
        except (KeyError, IndexError, AttributeError): pass

    for i, demo in enumerate(demos):
        obs, act = demo.get("observations"), demo.get("actions")
        # Basic type and dimension checks
        if not (isinstance(obs, np.ndarray) and isinstance(act, np.ndarray) and obs.ndim == 2 and act.ndim == 1): continue
        # Check expected dimension if known
        if expected_dim is not None and obs.shape[1] != expected_dim: continue
        # Check length consistency: obs = s0..sT, act = a0..aT-1 => len(obs) = len(act) + 1
        if len(obs) != len(act) + 1: continue

        # Extract states s0..s(T-1)
        states = obs[:-1]; actions = act;
        # Check for NaNs/Infs in the states we will use
        if np.isnan(states).any() or np.isinf(states).any(): continue

        # Store processed trajectory
        processed.append({
            "states": states.astype(np.float32),
            "actions": actions.astype(np.int64),
            "observations": obs.astype(np.float32) # Keep original observations too
        })
    print(f"Processed {len(demos)} raw demos into {len(processed)} valid trajectories for IRL.")
    return processed


# --- Main Execution Function ---
def example_usage(force_retrain_expert=False):
    """Example using MaxMargin IRL for LunarLander-v3 with PPO expert."""
    seed = 42 # Optional: for reproducibility
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = 'LunarLander-v3' # <<< Use v3 environment
    ppo_model_path = 'ppo_expert_lander.pth' # <<< Model path specific to v3

    main_env = None # Initialize main env variable
    try:
        main_env = gym.make(env_name)
        print(f"Created environment: {env_name}")
    except Exception as e:
        print(f"Error creating main environment {env_name}: {e}")
        print("Please ensure Gymnasium and Box2D are installed correctly ('pip install gymnasium[box2d]')")
        return None

    # --- Get/Train PPO Expert ---
    try:
        print("\n--- Setting up PPO Expert ---")
        ppo_expert = get_ppo_expert(main_env, device, force_retrain=force_retrain_expert, model_path=ppo_model_path)
        # Verify expert quality after loading/training
        min_acceptable_expert_score = 250 # Set a threshold for usable expert
        print("Evaluating loaded/trained PPO expert...")
        expert_eval_score, expert_eval_std = ppo_expert.evaluate()
        print(f"PPO Expert Eval Score: {expert_eval_score:.2f} +/- {expert_eval_std:.2f}")

        if expert_eval_score < min_acceptable_expert_score:
            print(f"\nPPO expert performance ({expert_eval_score:.2f}) is below required minimum ({min_acceptable_expert_score}).")
            print("IRL training is unlikely to succeed. Consider training PPO longer (--force_retrain_expert) or debugging.")
            if main_env: main_env.close();
            return None
    except Exception as e:
        print(f"Error during PPO expert setup: {e}");
        import traceback; traceback.print_exc(); # Print stack trace for debugging
        if main_env: main_env.close(); return None

    # --- Generate & Filter Demonstrations using PPO Expert ---
    try:
        print("\n--- Generating Expert Demonstrations ---")
        min_demo_reward_threshold = 250 # Filter out poor-performing demonstrations
        # Pass the expert instance explicitly
        expert_demos_raw, avg_kept_demo_reward = generate_expert_demos(
            expert_policy_ppo, main_env, n_demos=100, max_steps=1000,
            min_demo_reward=min_demo_reward_threshold, ppo_expert=ppo_expert
        )
        if not expert_demos_raw:
             print("Error: Failed to generate any expert demonstrations meeting the criteria.");
             if main_env: main_env.close(); return None
    except Exception as e:
        print(f"Error during demonstration generation: {e}")
        import traceback; traceback.print_exc();
        if main_env: main_env.close(); return None

    # --- Process Demonstrations for IRL ---
    expert_trajectories_processed = process_expert_demonstrations(expert_demos_raw)
    if not expert_trajectories_processed:
        print("Error: No valid expert trajectories after processing. Exiting.");
        if main_env: main_env.close(); return None

    print("\nProceeding with MaxMargin IRL training...\n")

    # --- Create and Train MaxMargin IRL ---
    maxmargin_irl = None # Initialize
    try:
        print("--- Initializing MaxMargin IRL ---")
        maxmargin_irl = MaxMarginIRL(
            env=main_env, # Use the main environment instance
            expert_trajectories=expert_trajectories_processed, # Use processed trajectories
            gamma=0.99,
            learning_rate_policy=5e-4, # Balanced learning rate
            n_policy_epochs=100,        # More epochs for better convergence
            policy_batch_size=256,      # Larger batch size for stability
            max_previous_mu=100,        # More history for better projection
            device=device
        )
    except ValueError as e:
        print(f"IRL Init Error: {e}")
        if main_env: main_env.close()
        return None
    except Exception as e:
        print(f"Unexpected IRL Init Error: {e}")
        import traceback
        traceback.print_exc()
        if main_env: main_env.close()
        return None


    # --- Train IRL ---
    results = None # Initialize
    try:
        print("\n--- Starting IRL Training ---")
        results = maxmargin_irl.train(
            n_iterations=100,           # More iterations for better convergence
            n_trajectories_per_iter=100, # More trajectories per iteration
            reward_threshold=200.0,      # Target eval reward for IRL policy
            eval_episodes=10,            # More evaluation episodes
            eval_interval=10             # Evaluate less frequently
        )
    except Exception as e:
        print(f"\n--- ERROR DURING IRL TRAINING ---")
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
        results = None # Indicate failure

    if not results:
        print("IRL Training failed or was interrupted.");
        if main_env: main_env.close(); return None

    # --- Final Evaluation & Output ---
    print("\n--- Final IRL Policy Evaluation ---")
    # Use the evaluate method from the trained IRL instance
    final_eval_reward, final_eval_std = maxmargin_irl.evaluate_policy(n_episodes=20)
    print(f"Final IRL Policy Eval Reward (20 episodes): {final_eval_reward:.2f} +/- {final_eval_std:.2f}")
    print(f"Best Average Eval Reward during training: {results.get('best_eval_reward', 'N/A'):.2f}")
    print(f"Final Learned Reward Weights:\n{results.get('reward_weights', 'N/A')}")

    # --- Render (Optional) ---
    render = False # <<< Set to True to watch the learned policy >>>
    if render and final_eval_reward > 100: # Only render if reasonable performance
        print("\nRendering learned IRL policy...")
        render_env = None
        try:
            render_env = gym.make(env_name, render_mode="human")
            for episode in range(3):
                state, _ = render_env.reset();
                if state is None or np.isnan(state).any() or np.isinf(state).any(): continue
                ep_reward = 0; done = False
                max_steps = render_env.spec.max_episode_steps if render_env.spec else 1000
                for _step in range(max_steps):
                    if state is None or np.isnan(state).any() or np.isinf(state).any(): break # Invalid state check
                    state_norm = maxmargin_irl.normalize_state(state)
                    if np.isnan(state_norm).any() or np.isinf(state_norm).any(): break # Invalid norm state check
                    state_tensor = torch.FloatTensor(state_norm).to(maxmargin_irl.device).unsqueeze(0)
                    action = maxmargin_irl.sample_action(state_tensor, deterministic=True, use_policy=True)
                    try:
                        next_state, reward, terminated, truncated, _ = render_env.step(action)
                        if next_state is None or np.isnan(next_state).any() or np.isinf(next_state).any():
                             done = True; next_state = state
                        else: done = terminated or truncated
                    except Exception: done = True; next_state = state
                    ep_reward += reward; state = next_state; render_env.render()
                    if done: break
                print(f"Rendered Episode {episode + 1} reward: {ep_reward}")
            if render_env: render_env.close()
        except Exception as e:
             print(f"Error during rendering: {e}")
             if render_env: render_env.close() # Ensure closure on error


    if main_env: main_env.close() # Close the original training env

    # --- Plotting ---
    if results and 'stats' in results:
        print("\nPlotting training statistics...")
        stats = results['stats']
        num_iterations_run = len(stats['policy_losses']) # Actual number of iterations completed
        iterations = np.arange(num_iterations_run)

        # Adjust eval iterations based on eval_interval and actual data length
        eval_interval = getattr(maxmargin_irl, 'eval_interval', 10)
        num_eval_points = len(stats['eval_rewards_mean'])
        # Iteration numbers corresponding to evaluations (starts at 0, then eval_interval, 2*eval_interval...)
        # Need to handle the case where training stops early.
        # Plot eval points against the iteration they occurred *after*. Initial is iter 0.
        # Subsequent evals happen after iter eval_interval, 2*eval_interval... up to num_iterations_run
        eval_iterations_axis = [0] + [min(eval_interval * k, num_iterations_run) for k in range(1, num_eval_points)]


        fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
        fig.suptitle(f'MaxMargin IRL Training - {env_name}', fontsize=16)

        # Eval Rewards
        try:
            mean_rewards = np.array(stats['eval_rewards_mean'])
            std_rewards = np.array(stats['eval_rewards_std'])
            # Ensure axis length matches data length
            valid_eval_iters_axis = eval_iterations_axis[:len(mean_rewards)]

            axs[0].plot(valid_eval_iters_axis, mean_rewards, label='Mean Eval Reward', marker='o', linestyle='-')
            axs[0].fill_between(valid_eval_iters_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label='Std Dev')
            axs[0].axhline(y=200, color='r', linestyle='--', label='Target (200)') # Target reward line
        except Exception as e: print(f"Plotting Eval Rewards Error: {e}")
        axs[0].set_ylabel("Avg Reward"); axs[0].set_title("IRL Policy Evaluation Performance"); axs[0].grid(True); axs[0].legend()

        # IRL Distance
        try:
             # Plot distance starting from iteration 1 (first update)
             if len(stats['irl_distances']) > 1:
                 axs[1].plot(iterations[1:], stats['irl_distances'][1:], label='IRL Distance (||mu_E - mu_bar||)', marker='.', linestyle='-')
        except Exception as e: print(f"Plotting IRL Distance Error: {e}")
        axs[1].set_ylabel("Distance"); axs[1].grid(True); axs[1].legend(); axs[1].set_yscale('log') # Log scale often useful

        # Policy Loss
        try:
             # Plot loss starting from iteration 1 (first optimization)
             if len(stats['policy_losses']) > 1:
                 axs[2].plot(iterations[1:], stats['policy_losses'][1:], label='Policy Loss', marker='.', linestyle='-')
        except Exception as e: print(f"Plotting Policy Loss Error: {e}")
        axs[2].set_ylabel("Loss"); axs[2].grid(True); axs[2].legend()

        # Weights Norm
        try:
            # Plot norm starting from iteration 1 (after first update)
             if len(stats['weights_norm']) > 1:
                 axs[3].plot(iterations[1:], stats['weights_norm'][1:], label='Reward Weights Norm ||w||', marker='.', linestyle='-')
        except Exception as e: print(f"Plotting Weights Norm Error: {e}")
        axs[3].set_ylabel("Norm"); axs[3].set_xlabel("Iteration"); axs[3].grid(True); axs[3].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
        # plt.show()

    return results


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MaxMargin IRL for LunarLander-v3')
    parser.add_argument('--force_retrain_expert', action='store_true', default=False,
                        help='Force retraining of the PPO expert policy, even if a saved model exists.')
    args = parser.parse_args()

    print("Starting MaxMargin IRL script for LunarLander-v3...")
    print(f"Force retrain expert: {args.force_retrain_expert}")

    start_time = time.time()
    final_results = example_usage(force_retrain_expert=args.force_retrain_expert)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"\nScript execution time: {total_time / 60:.2f} minutes ({total_time:.2f} seconds)")

    if final_results:
        print("\nScript completed successfully.")
    else:
        print("\nScript failed or exited early.")