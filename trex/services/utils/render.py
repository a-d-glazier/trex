"""
Utility functions for rendering environments and visualizing training results.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union
import torch

def render_episode(
    env: gym.Env,
    policy: torch.nn.Module,
    state_normalizer: callable,
    device: str,
    max_steps: int = 500,
    render_mode: str = "human"
) -> Dict[str, Union[float, List[float]]]:
    """
    Render a single episode using the provided policy.
    
    Args:
        env: Gymnasium environment
        policy: PyTorch policy network
        state_normalizer: Function to normalize states
        device: Device to run the policy on
        max_steps: Maximum number of steps per episode
        render_mode: Rendering mode for the environment
        
    Returns:
        Dictionary containing episode statistics
    """
    # Create environment with specified render mode
    render_env = gym.make(env.spec.id, render_mode=render_mode)
    
    state, _ = render_env.reset()
    episode_reward = 0
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        # Get action from policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_normalizer(state)).to(device)
            probs = policy(state_tensor.unsqueeze(0)).squeeze(0)
            action = torch.argmax(probs).item()
        
        # Execute action
        next_state, reward, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        state = next_state
        step_count += 1
    
    render_env.close()
    
    return {
        "reward": episode_reward,
        "steps": step_count
    }

def plot_training_stats(stats: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training statistics including rewards and losses.
    
    Args:
        stats: Dictionary containing training statistics
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot evaluation rewards
    ax1.plot(stats['eval_rewards'], label='Evaluation Reward')
    ax1.set_title('Evaluation Rewards Over Training')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Average Reward')
    ax1.grid(True)
    ax1.legend()
    
    # Plot losses
    ax2.plot(stats['reward_losses'], label='Reward Loss')
    ax2.plot(stats['policy_losses'], label='Policy Loss')
    ax2.set_title('Training Losses')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def render_multiple_episodes(
    env: gym.Env,
    policy: torch.nn.Module,
    state_normalizer: callable,
    device: str,
    n_episodes: int = 5,
    max_steps: int = 500,
    render_mode: str = "human"
) -> List[Dict[str, Union[float, List[float]]]]:
    """
    Render multiple episodes using the provided policy.
    
    Args:
        env: Gymnasium environment
        policy: PyTorch policy network
        state_normalizer: Function to normalize states
        device: Device to run the policy on
        n_episodes: Number of episodes to render
        max_steps: Maximum number of steps per episode
        render_mode: Rendering mode for the environment
        
    Returns:
        List of dictionaries containing episode statistics
    """
    episode_stats = []
    
    for episode in range(n_episodes):
        print(f"Rendering episode {episode + 1}/{n_episodes}")
        stats = render_episode(
            env=env,
            policy=policy,
            state_normalizer=state_normalizer,
            device=device,
            max_steps=max_steps,
            render_mode=render_mode
        )
        episode_stats.append(stats)
    
    return episode_stats 