import scipy.special
import numpy as np

def get_stochastic_policy(q):
    """
    Get the stochastic policy from the given Q-function.

    Args:
        q: A Q-function, i.e. a matrix of shape (n_states, n_actions).

    Returns:
        A stochastic policy, i.e. a matrix of shape (n_states, n_actions).
    """
    w = lambda x: x
    policy = scipy.special.softmax(w(q), 1)
    return policy, lambda state: np.random.choice([*range(policy.shape[1])], p=policy[state, :])

def get_deterministic_policy(q):
    return np.argmax(q, axis=1), lambda state: np.argmax(q[state, :])