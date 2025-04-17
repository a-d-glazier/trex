import numpy as np
from gym_simplegrid.envs.simple_grid import SimpleGridEnv
from gymnasium import spaces


class CustomSimpleGrid(SimpleGridEnv):
    def __init__(self, obstacle_map, render_mode="rgb_array"):
        super().__init__(obstacle_map, render_mode)
        self.n_states = self.nrow * self.ncol
        self.max_steps = 10
        # FOR GAIL
        # self.action_space = spaces.Box(0, 1, shape=(len(self.MOVES),), dtype=np.float64)
        self.observation_space = spaces.Box(0, 1, shape=(self.n_states,), dtype=np.float64)

    def seed(self, seed):
        # Only for compatibility; environment does not has any randomness
        np.random.seed(seed)

    def step(self, action: int):
        """
        Take a step in the environment.
        """
        #assert action in self.action_space
        self.agent_action = action
        action_idx = action #np.argmax(action)

        # Get the current position of the agent
        row, col = self.agent_xy
        dx, dy = self.MOVES[action_idx]

        # Compute the target position of the agent
        target_row = row + dx
        target_col = col + dy

        # Compute the reward
        self.reward = self.get_reward(target_row, target_col)
        
        # Check if the move is valid
        if self.is_in_bounds(target_row, target_col) and self.is_free(target_row, target_col):
            self.agent_xy = (target_row, target_col)
            self.done = self.on_goal()

        self.n_iter += 1

        # Check for maximum episode length
        if self.n_iter >= self.max_steps:
            self.done = True

        # if self.render_mode == "human":
        self.render()

        return self.get_obs(), self.reward, self.done, False, self.get_info()

    def get_obs(self):
        """Return the one-hot encoded state representation."""
        obs = np.zeros(self.n_states)
        state_idx = self.agent_xy[0] * self.ncol + self.agent_xy[1]
        obs[state_idx] = 1
        return obs


    # def to_s(self, row: int, col: int) -> int: # for GAIL
    #     return np.eye(self.n_states)[row * self.ncol + col]

    def reset(self, seed=None, options=None):
        if not options:
            options = {'start_loc': 0, 'goal_loc': 8}
        return super().reset(seed=seed, options=options)
    
    def get_reward(self, x: int, y: int) -> float:
        """
        Get the reward of a given cell.
        """
        if not self.is_in_bounds(x, y):
            return -1.0
        elif not self.is_free(x, y):
            return -1.0
        elif (x, y) == self.goal_xy:
            return 10.0
        else:
            return -1.0

    def get_transition_probabilities(self) -> np.ndarray:
        """
        Computes the transition probability matrix.

        The transition probability matrix is a 3D NumPy array `P` where:
        - `P[s, s', a]` gives the probability of transitioning from state `s` to state `s'` given action `a`.

        Returns
        -------
        np.ndarray
            A 3D array of shape (n_states, n_states, n_actions) representing the transition probabilities.
        """
        n_states = self.nrow * self.ncol
        n_actions = len(self.MOVES)
        P = np.zeros((n_states, n_states, n_actions))

        for s in range(n_states):
            row, col = self.to_xy(s)
            if not self.is_free(row, col):
                continue  # No transitions from obstacle states

            for a, (dx, dy) in self.MOVES.items():
                target_row, target_col = row + dx, col + dy
                if self.is_in_bounds(target_row, target_col) and self.is_free(target_row, target_col):
                    s_prime = self.to_s(target_row, target_col)
                else:
                    s_prime = s  # Stay in place if hitting a wall or boundary

                P[s, s_prime, a] = 1.0  # Deterministic transitions

        return P

    def get_reward_function(self) -> np.ndarray:
        """
        Returns the reward function as a NumPy array of shape `(num_states, num_actions, num_states)`,
        where `R[s, a, s_prime]` is the reward for taking action `a` in state `s` and ending up in `s_prime`.
        """
        num_states = self.nrow * self.ncol
        num_actions = len(self.MOVES)
        R = np.zeros((num_states, num_actions, num_states))  # Shape (S, A, S')

        for s in range(num_states):
            row, col = self.to_xy(s)  # Convert state index to (row, col)
            for a in range(num_actions):
                dx, dy = self.MOVES[a]
                target_row, target_col = row + dx, col + dy  # Compute next state
                
                # Check if next state is valid
                if self.is_in_bounds(target_row, target_col) and self.is_free(target_row, target_col):
                    s_prime = self.to_s(target_row, target_col)  # Convert (row, col) to state index
                    next_row, next_col = target_row, target_col
                else:
                    s_prime = s  # If invalid, stay in the same state
                    next_row, next_col = row, col

                R[s, a, s_prime] = self.get_reward(next_row, next_col)  # Reward based on (s, a, s')

        return R

    def get_feature_matrix(self):
        """
        Return the feature matrix assigning each state with an individual
        feature (i.e. an identity matrix of size n_states * n_states).

        Rows represent individual states, columns the feature entries.

        Args:
            world: A GridWorld instance for which the feature-matrix should be
                computed.

        Returns:
            The coordinate-feature-matrix for the specified world.
        """
        return np.identity(self.n_states)