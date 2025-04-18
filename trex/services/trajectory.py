import numpy as np
from collections import namedtuple

Demonstration = namedtuple('Demonstration', ['trajectories', 'policy'])

class Trajectory:
    """
    A trajectory consisting of states, corresponding actions, and outcomes.

    Args:
        transitions: The transitions of this trajectory as an array of
            tuples `(state_from, action, state_to)`. Note that `state_to` of
            an entry should always be equal to `state_from` of the next
            entry.
    """

    def __init__(self, transitions):
        self._t = list(transitions)

    def transitions(self):
        """
        The transitions of this trajectory.

        Returns:
            All transitions in this trajectory as array of tuples
            `(state_from, action, state_to)`.
        """
        return list(self._t)

    def __repr__(self):
        return "Trajectory({})".format(repr(self._t))

    def __str__(self):
        return "{}".format(self._t)

def generate_trajectory(world, policy, start, final, max_len=200):
    """
    Generate a single trajectory.

    Args:
        world: The world for which the trajectory should be generated.
        policy: A function (state: Integer) -> (action: Integer) mapping a
            state to an action, specifying which action to take in which
            state. This function may return different actions for multiple
            invokations with the same state, i.e. it may make a
            probabilistic decision and will be invoked anew every time a
            (new or old) state is visited (again).
        start: The starting state (as Integer index).
        final: A collection of terminal states. If a trajectory reaches a
            terminal state, generation is complete and the trajectory is
            returned.

    Returns:
        A generated Trajectory instance adhering to the given arguments.
    """

    state = start

    trajectory = []
    trial = 0
    transition_probabilities = world.unwrapped.get_transition_probabilities()
    while state not in [final]:
        if len(trajectory) > max_len:  # Reset and create a new trajectory
            if trial >= 5:
                # print('Warning: terminated trajectory generation due to unreachable final state.')
                return Trajectory(trajectory), False    #break
            trajectory = []
            state = start
            trial += 1

        action = policy(state)

        next_s = range(world.unwrapped.n_states)
        next_p = transition_probabilities[state, :, action]

        next_state = np.random.choice(next_s, p=next_p)

        trajectory.append((state, action, next_state))
        state = next_state

    return Trajectory(trajectory), True

def generate_trajectories(n, world, policy, start, final, discard_not_feasable=False):
    """
    Generate multiple trajectories.

    Args:
        n: The number of trajectories to generate.
        world: The world for which the trajectories should be generated.
        policy: A function `(state: Integer) -> action: Integer` mapping a
            state to an action, specifying which action to take in which
            state. This function may return different actions for multiple
            invokations with the same state, i.e. it may make a
            probabilistic decision and will be invoked anew every time a
            (new or old) state is visited (again).
        start: The starting state (as Integer index), a list of starting
            states (with uniform probability), or a list of starting state
            probabilities, mapping each state to a probability. Iff the
            length of the provided list is equal to the number of states, it
            is assumed to be a probability distribution over all states.
            Otherwise it is assumed to be a list containing all starting
            state indices, an individual state is then chosen uniformly.
        final: A collection of terminal states. If a trajectory reaches a
            terminal state, generation is complete and the trajectory is
            complete.
        discard_not_feasable: Discard trajectories that not reaching the 
            final state(s)

    Returns:
        A generator expression generating `n` `Trajectory` instances
        adhering to the given arguments.
    """
    start_states = np.atleast_1d(start)

    def _generate_one():
        if len(start_states) == world.unwrapped.n_states:
            s = np.random.choice(range(world.unwrapped.n_states), p=start_states)
        else:
            s = np.random.choice(start_states)

        return generate_trajectory(world, policy, s, final)

    list_tr = []
    for _ in range(n):
        tr, reachable = _generate_one()
        if reachable or not discard_not_feasable:
            list_tr.append(tr)
    
    return list_tr

def generate_demonstrations(world, policy, start, terminal, n_trajectories=200):
    """
    Generate some "expert" trajectories.
    """
    # parameters
    discount = 0.9

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.unwrapped.n_states)
    initial[start] = 1.0

    # generate trajectories
    policy_exec = lambda state: np.random.choice([*range(policy.shape[1])], p=policy[state, :])
    tjs = generate_trajectories(n_trajectories, world, policy_exec, initial, terminal)

    if not tjs:
        return False
    return Demonstration(tjs, policy)

def rank_trajectories_by_reward(trajectories, env):
    """
    Rank the given trajectories using the reward function.

    Args:
        trajectories: A list of `Trajectory` instances to rank.
        env: The environment in which the trajectories were generated.

    Returns:
        A list of tuples `(trajectory, score)` where `score` is the result
        of the ranking function applied to the trajectory.
    """
    result = []
    for trajectory in trajectories:
        trajectory_score = 0
        for state, action, next_state in trajectory.transitions():
            next_state = env.unwrapped.to_xy(next_state)
            trajectory_score += env.unwrapped.get_reward(*next_state)
        result.append((trajectory, trajectory_score))

    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    return sorted_result