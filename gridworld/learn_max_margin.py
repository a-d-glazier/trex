import gymnasium as gym
from envs import *
import numpy as np

from services.rl.value_iteration import value_iteration
from services.rl.policy import get_stochastic_policy
from services.trajectory import generate_demonstrations
from services.irl.max_margin import irl

np.random.seed(7)

env = gym.make(
    'DummyGrid-v0', 
    render_mode='human'
)
observations, info = env.reset(options={'start_loc':0, 'goal_loc':8})

# Get expert policy
v, q, _ = value_iteration(
    env.unwrapped.get_transition_probabilities(), 
    env.unwrapped.get_reward_function(), 
    discount=0.9,
)
policy, policy_exec = get_stochastic_policy(q)

# Generate expert demonstrations
trajectories = generate_demonstrations(env, policy, 0, 8, 200)

reward = irl(
    env=env,
    p_transition=env.unwrapped.get_transition_probabilities(),
    feature_matrix=env.unwrapped.get_feature_matrix(),
    expert_trajectories=trajectories[0],
)

def get_reward_matrix(reward, num_states, num_actions):
    R = np.zeros((num_states, num_actions, num_states))
    for s in range(num_states):
        for a in range(num_actions):
            for s_prime in range(num_states):
                R[s, a, s_prime] = reward[s]
    return R

R = get_reward_matrix(reward, env.unwrapped.n_states, len(env.unwrapped.MOVES)) 
v, q , _ = value_iteration(
    env.unwrapped.get_transition_probabilities(), 
    R,
    discount=0.9
)

env.reset(options={'start_loc':0, 'goal_loc':8})
for _ in range(100):
    action = policy_exec(observations)
    observations, reward, done, _, info = env.step(action)
    if done:
        observations, info = env.reset(options={'start_loc':0, 'goal_loc':8})