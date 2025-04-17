import numpy as np
from services.rl.policy import get_stochastic_policy
from services.trajectory import generate_demonstrations, rank_trajectories_by_reward

def value_iteration(p, reward, discount, eps=1e-4, checkpoint_trajectories=False, env=None):
    """
    Basic value-iteration algorithm to solve the given MDP.

    Args:
        p: A NumPy array of shape (num_states, num_states, num_actions) representing
           transition probabilities, where p[s, s', a] is the probability of transitioning
           from state `s` to state `s'` by taking action `a`.
        reward: A NumPy array of shape (num_states, num_actions) representing rewards,
                where reward[s, a] gives the reward for taking action `a` in state `s`.
        discount: The discount factor (gamma) applied during value iteration.
        eps: Convergence threshold. The algorithm stops when the value function changes
             less than `eps` for all states in an iteration.

    Returns:
        V: A NumPy array of shape (num_states,) containing the optimal state values.
        Q: A NumPy array of shape (num_states, num_actions) containing the optimal action values.
    """
    num_states, _, num_actions = p.shape
    
    # Initialize V and Q to zero
    V = np.zeros(num_states)
    Q = np.zeros((num_states, num_actions))

    all_trajectories = []
    i = 0
    while True:
        i += 1
        delta = 0
        new_V = np.zeros(num_states)  # Store new values of V

        for s in range(num_states):
            for a in range(num_actions):
                # Compute Q-value for (s, a)
                Q[s, a] = np.sum(p[s, :, a] * (reward[s, a, :] + discount * V))

            # Update V(s) as max Q(s, a)
            new_V[s] = np.max(Q[s])

            # Track largest change for convergence
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V  # Update V

        if checkpoint_trajectories:
            if i < 5: # fix this
                policy, policy_exec = get_stochastic_policy(Q)
                trajectories = generate_demonstrations(env, policy, 0, 8, 200)
                all_trajectories.append(trajectories)

        # Check convergence
        if delta < eps:
            break

    return V, Q, all_trajectories