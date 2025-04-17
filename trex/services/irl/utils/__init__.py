import numpy as np

def feature_expectation_from_trajectories(features, trajectories):
    """
    Compute the feature expectation of the given trajectories.

    Simply counts the number of visitations to each feature-instance and
    divides them by the number of trajectories.

    Args:
        features: The feature-matrix (e.g. as numpy array), mapping states
            to features, i.e. a matrix of shape (n_states x n_features).
        trajectories: A list or iterator of `Trajectory` instances.

    Returns:
        The feature-expectation of the provided trajectories as map
        `[state: Integer] -> feature_expectation: Float`.
    """
    n_states, n_features = features.shape

    fe = np.zeros(n_features)

    for t in trajectories:
        for s, a, s_prime in t.transitions():
            fe += features[s, :]

    return fe / len(trajectories)