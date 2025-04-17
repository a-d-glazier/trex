import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Union
import gymnasium as gym
from tqdm import tqdm
import os

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

class Discriminator(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Ensure action is one-hot encoded
        if action.dim() == 1:
            action = torch.nn.functional.one_hot(action, num_classes=2).float()
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class Policy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Policy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)
    
    def get_greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        probs = self.forward(state)
        return torch.argmax(probs)

class GAIL:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-4):
        self.policy = Policy(state_dim, action_dim)
        self.discriminator = Discriminator(state_dim, action_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        self.gamma = 0.99
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.discriminator.to(self.device)
        self.entropy_coef = 0.01
    
    def compute_returns(self, rewards: List[float], dones: List[bool]) -> torch.Tensor:
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update_discriminator(self, expert_states: torch.Tensor, expert_actions: torch.Tensor,
                           generated_states: torch.Tensor, generated_actions: torch.Tensor) -> float:
        expert_preds = self.discriminator(expert_states, expert_actions)
        generated_preds = self.discriminator(generated_states, generated_actions)
        
        expert_labels = torch.ones_like(expert_preds, device=self.device)
        generated_labels = torch.zeros_like(generated_preds, device=self.device)
        
        discriminator_loss = nn.BCELoss()(expert_preds, expert_labels) + \
                           nn.BCELoss()(generated_preds, generated_labels)
        
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
        self.discriminator_optimizer.step()
        
        return discriminator_loss.item()
    
    def update_policy(self, states: torch.Tensor, actions: torch.Tensor,
                     log_probs: torch.Tensor, returns: torch.Tensor) -> float:
        # Compute discriminator rewards
        with torch.no_grad():
            discriminator_rewards = torch.log(self.discriminator(states, actions) + 1e-8)
        
        # Compute policy loss with entropy bonus
        probs = self.policy(states)
        dist = Categorical(probs)
        entropy = dist.entropy().mean()
        
        policy_loss = -(log_probs * (returns + discriminator_rewards.squeeze())).mean() - self.entropy_coef * entropy
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()
        
        return policy_loss.item()

def collect_trajectory(env: gym.Env, policy: Union[Policy, callable], device: torch.device, is_expert: bool = False) -> Dict[str, torch.Tensor]:
    states, actions, rewards, dones, log_probs = [], [], [], [], []
    state, _ = env.reset()
    done = False
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        if is_expert:
            action = policy(state)  # Call expert_policy function directly
        else:
            action, log_prob = policy.get_action(state_tensor)
            log_probs.append(log_prob)
        
        next_state, reward, terminated, truncated, _ = env.step(action if is_expert else action.item())
        done = terminated or truncated
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        
        state = next_state
    
    trajectory = {
        'states': torch.FloatTensor(states).to(device),
        'actions': torch.LongTensor(actions).to(device),
        'rewards': torch.FloatTensor(rewards).to(device),
        'dones': torch.FloatTensor(dones).to(device)
    }
    
    if not is_expert:
        trajectory['log_probs'] = torch.stack(log_probs)
    
    return trajectory

def collect_expert_trajectories(env: gym.Env, num_trajectories: int = 100) -> List[Dict[str, torch.Tensor]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trajectories = []
    
    for i in tqdm(range(num_trajectories), desc="Collecting Expert Trajectories"):
        trajectory = collect_trajectory(env, expert_policy, device, is_expert=True)
        trajectories.append(trajectory)
    
    return trajectories

def main():
    # Create environment
    env = gym.make('CartPole-v1', render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Collect expert trajectories using PID-like expert policy
    print("Collecting expert trajectories...")
    expert_trajectories = collect_expert_trajectories(env)
    torch.save(expert_trajectories, 'expert_trajectories.pt')
    
    # Initialize GAIL
    print("Starting GAIL training...")
    gail = GAIL(state_dim, action_dim)
    
    # Prepare expert data
    expert_states = torch.cat([t['states'] for t in expert_trajectories])
    expert_actions = torch.cat([t['actions'] for t in expert_trajectories])
    
    # Training loop
    num_episodes = 2000
    best_reward = -float('inf')
    reward_threshold = 450  # Target reward to achieve
    eval_episodes = 10  # Number of episodes to average for evaluation
    
    for episode in tqdm(range(num_episodes), desc="GAIL Training"):
        # Collect trajectory
        trajectory = collect_trajectory(env, gail.policy, gail.device)
        
        # Compute returns
        returns = gail.compute_returns(trajectory['rewards'].tolist(), trajectory['dones'].tolist())
        
        # Update discriminator
        discriminator_loss = gail.update_discriminator(
            expert_states, expert_actions,
            trajectory['states'], trajectory['actions']
        )
        
        # Update policy
        policy_loss = gail.update_policy(
            trajectory['states'],
            trajectory['actions'],
            trajectory['log_probs'],
            returns
        )
        
        # Evaluate policy
        if episode % 10 == 0:
            eval_rewards = []
            for _ in range(eval_episodes):
                state, _ = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(gail.device)
                    action = gail.policy.get_greedy_action(state_tensor)
                    next_state, reward, terminated, truncated, _ = env.step(action.item())
                    done = terminated or truncated
                    episode_reward += reward
                    state = next_state
                
                eval_rewards.append(episode_reward)
            
            avg_reward = np.mean(eval_rewards)
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(gail.policy.state_dict(), 'gail_cartpole.pth')
                
                # Check if we've reached the reward threshold
                if best_reward >= reward_threshold:
                    print(f"\nReached reward threshold of {reward_threshold}. Early stopping.")
                    break
            
            print(f"Episode {episode}, Avg Eval Reward: {avg_reward:.2f} ± {np.std(eval_rewards):.2f}, "
                  f"Discriminator Loss: {discriminator_loss:.4f}, "
                  f"Policy Loss: {policy_loss:.4f}")
    
    # Final evaluation with rendering
    print("\nFinal Evaluation of Learned Policy:")
    env_render = gym.make('CartPole-v1', render_mode="human")
    
    # Load best policy
    gail.policy.load_state_dict(torch.load('gail_cartpole.pth'))
    gail.policy.eval()
    
    eval_rewards = []
    num_eval_episodes = 3
    
    print(f"\nRendering {num_eval_episodes} episodes with the learned policy...")
    
    for episode in range(num_eval_episodes):
        state, _ = env_render.reset()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(gail.device)
                action = gail.policy.get_greedy_action(state_tensor)
            
            next_state, reward, terminated, truncated, _ = env_render.step(action.item())
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            
            env_render.render()
        
        eval_rewards.append(episode_reward)
        print(f"Episode {episode + 1} reward: {episode_reward}")
    
    env_render.close()
    env.close()
    
    print(f"\nAverage evaluation reward: {np.mean(eval_rewards):.2f}")
    print(f"Standard deviation: {np.std(eval_rewards):.2f}")
    print(f"Best evaluation reward: {max(eval_rewards):.2f}")
    print(f"Worst evaluation reward: {min(eval_rewards):.2f}")

if __name__ == "__main__":
    main() 