import numpy as np
import time
from .utils import feature_expectation_from_trajectories
from ..rl.value_iteration import value_iteration
from ..rl.policy import get_stochastic_policy
from ..trajectory import generate_demonstrations

def get_reward_matrix(reward, num_states, num_actions):
    R = np.zeros((num_states, num_actions, num_states))
    for s in range(num_states):
        for a in range(num_actions):
            for s_prime in range(num_states):
                R[s, a, s_prime] = reward[s]
    return R

def irl(env, p_transition, feature_matrix, expert_trajectories, epsilon = 0.1):
    '''
    Generate the expert's trajectories according to the optimal policy
    and compute the feature expectations. 
    '''   


    time_start = time.time()
    n_trajectories = len(expert_trajectories)
    n_states, d_states = feature_matrix.shape
    expert_feature_expectations = feature_expectation_from_trajectories(feature_matrix, expert_trajectories)
    
    ''' 
    Pick the initial policy
    generate the trajectories according to the initial policy
    compute the feature expectations
    '''                      
    # np.random.seed(10)
    w = np.random.uniform(size=(d_states,))
    R = feature_matrix.dot(w)
    R = get_reward_matrix(R, n_states, len(env.unwrapped.MOVES))

    _, Q_table, _ = value_iteration(
        p_transition,
        R,
        discount=0.9,
    )
    policy, policy_exec = get_stochastic_policy(Q_table)
    updated_policy_trajectories = generate_demonstrations(env, policy, 0, 8, n_trajectories)
    feature_expectations_bar = feature_expectation_from_trajectories(feature_matrix, updated_policy_trajectories[0])
    

    '''
    while loop for policy iteration: In this loop, we apply the computation trick in section 3.1 of 
    Ng & Abeel's paper. E.g. the projection margin method.
    '''
    w = expert_feature_expectations - feature_expectations_bar
    t = np.linalg.norm(w, 2)
    print("Initial threthod: ", t)
    i = 0

    while t > epsilon:

        R = feature_matrix.dot(w)
        R = get_reward_matrix(R, n_states, len(env.unwrapped.MOVES))

        _, Q_table, _ = value_iteration(
            p_transition,
            R,
            discount=0.9,
        )
        policy, policy_exec = get_stochastic_policy(Q_table)
        updated_policy_trajectories = generate_demonstrations(env, policy, 0, 8, n_trajectories)
        feature_expectations = feature_expectation_from_trajectories(feature_matrix, updated_policy_trajectories[0])
        updated_loss = feature_expectations-feature_expectations_bar
        feature_expectations_bar += updated_loss*updated_loss.dot(w)/np.square(updated_loss).sum()
        w = expert_feature_expectations-feature_expectations_bar
        t = np.linalg.norm(w, 2)
        i += 1

        #print distance t every 100 iterations. 
        if i % 100 == 0: 
            print('The '+ str(i) +'th threshold is '+ str(t)+'.')

    time_elasped = time.time() -  time_start
    print('Total Apprenticeship computational time is /n', time_elasped)
       

    return feature_matrix.dot(w).reshape((n_states,))