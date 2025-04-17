# See: https://github.com/qzed/irl-maxent/blob/master/src/

import numpy as np
from .utils import feature_expectation_from_trajectories

def compute_expected_svf(p_transition, p_initial, terminal, reward, eps=1e-5):
    """
    Compute the expected state visitation frequency for maximum entropy IRL.

    This is an implementation of Algorithm 1 of the Maximum Entropy IRL
    paper by Ziebart et al. (2008).

    This function combines the backward pass implemented in
    `local_action_probabilities` with the forward pass implemented in
    `expected_svf_from_policy`.

    Args:
        p_transition: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        p_initial: The probability of a state being an initial state as map
            `[state: Integer] -> probability: Float`.
        terminal: A list of terminal states.
        reward: The reward signal per state as table
            `[state: Integer] -> reward: Float`.
        eps: The threshold to be used as convergence criterion for the
            expected state-visitation frequency. Convergence is assumed if
            the expected state visitation frequency changes less than the
            threshold on all states in a single iteration.

    Returns:
        The expected state visitation frequencies as map
        `[state: Integer] -> svf: Float`.
    """
    p_action = local_action_probabilities(p_transition, terminal, reward)
    return expected_svf_from_policy(p_transition, p_initial, terminal, p_action, eps)

def expected_svf_from_policy(p_transition, p_initial, terminal, p_action, eps=1e-5):
    """
    Compute the expected state visitation frequency using the given local
    action probabilities.

    This is the forward pass of Algorithm 1 of the Maximum Entropy IRL paper
    by Ziebart et al. (2008). Alternatively, it can also be found as
    Algorithm 9.3 in in Ziebart's thesis (2010).

    It has been slightly adapted for convergence, by forcing transition
    probabilities from terminal stats to be zero.

    Args:
        p_transition: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        p_initial: The probability of a state being an initial state as map
            `[state: Integer] -> probability: Float`.
        terminal: A list of terminal states.
        p_action: Local action probabilities as map
            `[state: Integer, action: Integer] -> probability: Float`
            as returned by `local_action_probabilities`.
        eps: The threshold to be used as convergence criterion. Convergence
            is assumed if the expected state visitation frequency changes
            less than the threshold on all states in a single iteration.

    Returns:
        The expected state visitation frequencies as map
        `[state: Integer] -> svf: Float`.
    """
    n_states, _, n_actions = p_transition.shape

    # 'fix' our transition probabilities to allow for convergence
    # we will _never_ leave any terminal state
    p_transition = np.copy(p_transition)
    p_transition[terminal, :, :] = 0.0

    # set-up transition matrices for each action
    p_transition = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

    # actual forward-computation of state expectations
    d = np.zeros(n_states)

    delta = np.inf
    while delta > eps:
        d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
        d_ = p_initial + np.array(d_).sum(axis=0)

        delta, d = np.max(np.abs(d_ - d)), d_

    return d


# -- plain maximum entropy (Ziebart et al. 2008) -------------------------------

def local_action_probabilities(p_transition, terminal, reward):
    """
    Compute the local action probabilities (policy) required for the edge
    frequency calculation for maximum entropy reinfocement learning.

    This is the backward pass of Algorithm 1 of the Maximum Entropy IRL
    paper by Ziebart et al. (2008).

    Args:
        p_transition: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        terminal: A set/list of terminal states.
        reward: The reward signal per state as table
            `[state: Integer] -> reward: Float`.

    Returns:
        The local action probabilities (policy) as map
        `[state: Integer, action: Integer] -> probability: Float`
    """
    n_states, _, n_actions = p_transition.shape

    er = np.exp(reward)
    p = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

    # initialize at terminal states
    zs = np.zeros(n_states)
    zs[terminal] = 1.0

    # perform backward pass
    # This does not converge, instead we iterate a fixed number of steps. The
    # number of steps is chosen to reflect the maximum steps required to
    # guarantee propagation from any state to any other state and back in an
    # arbitrary MDP defined by p_transition.
    for _ in range(2 * n_states):
        za = np.array([er * p[a].dot(zs) for a in range(n_actions)]).T
        zs = za.sum(axis=1)

    # compute local action probabilities
    return za / zs[:, None]

def initial_probabilities_from_trajectories(n_states, trajectories):
    """
    Compute the probability of a state being a starting state using the
    given trajectories.

    Args:
        n_states: The number of states.
        trajectories: A list or iterator of `Trajectory` instances.

    Returns:
        The probability of a state being a starting-state as map
        `[state: Integer] -> probability: Float`.
    """
    p = np.zeros(n_states)

    for t in trajectories:
        p[t.transitions()[0][0]] += 1.0

    return p / len(trajectories)

class Initializer:
    """
    Base-class for an Initializer, specifying a strategy for parameter
    initialization.
    """
    def __init__(self):
        pass

    def initialize(self, shape):
        """
        Create an initial set of parameters.

        Args:
            shape: The shape of the parameters.

        Returns:
            An initial set of parameters of the given shape, adhering to the
            initialization-strategy described by this Initializer.
        """
        raise NotImplementedError

    def __call__(self, shape):
        """
        Create an initial set of parameters.

        Note:
            This function simply calls `self.initialize(shape)`.

        Args:
            shape: The shape of the parameters.

        Returns:
            An initial set of parameters of the given shape, adhering to the
            initialization-strategy described by this Initializer.
        """
        return self.initialize(shape)

class Constant(Initializer):
    """
    An Initializer, initializing parameters to a constant value.

    Args:
        value: Either a scalar value or a function in dependence on the
            shape of the parameters, returning a scalar value for
            initialization.
    """
    def __init__(self, value=1.0):
        super().__init__()
        self.value = value

    def initialize(self, shape):
        """
        Create set of parameters with initial fixed value.

        The scalar value used for initialization can be specified in the
        constructor.

        Args:
            shape: The shape of the parameters.

        Returns:
            An set of constant-valued parameters of the given shape.
        """
        if callable(self.value):
            return np.ones(shape) * self.value(shape)
        else:
            return np.ones(shape) * self.value

def linear_decay(lr0=0.2, decay_rate=1.0, decay_steps=1):
    """
    Linear learning-rate decay.

    Creates a function `(k: Integer) -> learning_rate: Float` returning the
    learning-rate in dependence on the current number of iterations. The
    returned function can be expressed as

        learning_rate(k) = lr0 / (1.0 + decay_rate * floor(k / decay_steps))

    Args:
        lr0: The initial learning-rate.
        decay_rate: The decay factor.
        decay_steps: An integer number of steps that can be used to
            staircase the learning-rate.

    Returns:
        The function giving the current learning-rate in dependence of the
        current iteration as specified above.
    """
    def _lr(k):
        return lr0 / (1.0 + decay_rate * np.floor(k / decay_steps))

    return _lr

class Optimizer:
    """
    Optimizer base-class.

    Note:
        Before use of any optimizer, its `reset` function must be called.

    Attributes:
        parameters: The parameters to be optimized. This should only be set
            via the `reset` method of this optimizer.
    """
    def __init__(self):
        self.parameters = None

    def reset(self, parameters):
        """
        Reset this optimizer.

        Args:
            parameters: The parameters to optimize.
        """
        self.parameters = parameters

    def step(self, grad, *args, **kwargs):
        """
        Perform a single optimization step.

        Args:
            grad: The gradient used for the optimization step.

            Other arguments are optimizer-specific.
        """
        raise NotImplementedError

    def normalize_grad(self, ord=None):
        """
        Create a new wrapper for this optimizer which normalizes the
        gradient before each step.

        Returns:
            An Optimizer instance wrapping this Optimizer, normalizing the
            gradient before each step.

        See also:
            `class NormalizeGrad`
        """
        return NormalizeGrad(self, ord)

class NormalizeGrad(Optimizer):
    """
    A wrapper wrapping another Optimizer, normalizing the gradient before
    each step.

    For every call to `step`, this Optimizer will normalize the gradient and
    then pass the normalized gradient on to the underlying optimizer
    specified in the constructor.

    Note:
        Before use of any optimizer, its `reset` function must be called.

    Args:
        opt: The underlying optimizer to be used.
        ord: The order of the norm to be used for normalizing. This argument
            will be direclty passed to `numpy.linalg.norm`.
    """
    def __init__(self, opt, ord=None):
        super().__init__()
        self.opt = opt
        self.ord = ord

    def reset(self, parameters):
        """
        Reset this optimizer.

        Args:
            parameters: The parameters to optimize.
        """
        super().reset(parameters)
        self.opt.reset(parameters)

    def step(self, grad, *args, **kwargs):
        """
        Perform a single optimization step.

        This will call the underlying optimizer with the normalized
        gradient.

        Args:
            grad: The gradient used for the optimization step.

            Other arguments depend on the underlying optimizer.
        """
        return self.opt.step(grad / np.linalg.norm(grad, self.ord), *args, **kwargs)

class ExpSga(Optimizer):
    """
    Exponentiated stochastic gradient ascent.

    The implementation follows Algorithm 10.5 from B. Ziebart's thesis
    (2010) and is slightly adapted from the original algorithm provided by
    Kivinen and Warmuth (1997).

    Note:
        Before use of any optimizer, its `reset` function must be called.

    Args:
        lr: The learning-rate. This may either be a float for a constant
            learning-rate or a function
            `(k: Integer) -> learning_rate: Float`
            taking the step number as parameter and returning a learning
            rate as result.
            See also `linear_decay`, `power_decay` and `exponential_decay`.
        normalize: A boolean specifying if the the parameters should be
            normalized after each step, as done in the original algorithm by
            Kivinen and Warmuth (1997).

    Attributes:
        parameters: The parameters to be optimized. This should only be set
            via the `reset` method of this optimizer.
        lr: The learning-rate as specified in the __init__ function.
        k: The number of steps run since the last reset.
    """
    def __init__(self, lr, normalize=False):
        super().__init__()
        self.lr = lr
        self.normalize = normalize
        self.k = 0

    def reset(self, parameters):
        """
        Reset this optimizer.

        Args:
            parameters: The parameters to optimize.
        """
        super().reset(parameters)
        self.k = 0

    def step(self, grad, *args, **kwargs):
        """
        Perform a single optimization step.

        Args:
            grad: The gradient used for the optimization step.
        """
        lr = self.lr if not callable(self.lr) else self.lr(self.k)
        self.k += 1

        self.parameters *= np.exp(lr * grad)

        if self.normalize:
            self.parameters /= self.parameters.sum()

def irl(p_transition, features, terminal, trajectories, optim=ExpSga(lr=linear_decay(lr0=0.2)), init=Constant(1.0), eps=1e-4, eps_esvf=1e-5):
    """
    Compute the reward signal given the demonstration trajectories using the
    maximum entropy inverse reinforcement learning algorithm proposed in the
    corresponding paper by Ziebart et al. (2008).

    Args:
        p_transition: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        features: The feature-matrix (e.g. as numpy array), mapping states
            to features, i.e. a matrix of shape (n_states x n_features).
        terminal: A list of terminal states.
        trajectories: A list of `Trajectory` instances representing the
            expert demonstrations.
        optim: The `Optimizer` instance to use for gradient-based
            optimization.
        init: The `Initializer` to use for initialization of the reward
            function parameters.
        eps: The threshold to be used as convergence criterion for the
            reward parameters. Convergence is assumed if all changes in the
            scalar parameters are less than the threshold in a single
            iteration.
        eps_svf: The threshold to be used as convergence criterion for the
            expected state-visitation frequency. Convergence is assumed if
            the expected state visitation frequency changes less than the
            threshold on all states in a single iteration.

    Returns:
        The reward per state as table `[state: Integer] -> reward: Float`.
    """
    n_states, _, n_actions = p_transition.shape
    _, n_features = features.shape

    # compute static properties from trajectories
    e_features = feature_expectation_from_trajectories(features, trajectories)
    p_initial = initial_probabilities_from_trajectories(n_states, trajectories)

    # basic gradient descent
    theta = init(n_features)
    delta = np.inf

    optim.reset(theta)
    while delta > eps:
        theta_old = theta.copy()

        # compute per-state reward
        reward = features.dot(theta)

        # compute the gradient
        e_svf = compute_expected_svf(p_transition, p_initial, terminal, reward, eps_esvf)
        grad = e_features - features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)
        delta = np.max(np.abs(theta_old - theta))

    # re-compute per-state reward and return
    return features.dot(theta)