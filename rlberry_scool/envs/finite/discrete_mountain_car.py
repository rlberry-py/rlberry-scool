import numpy as np
import gymnasium as gym


def binary_search_nd(x_vec, bins):
    """n-dimensional binary search

    Parameters
    -----------
    x_vec : numpy.ndarray
        numpy 1d array to be searched in the bins
    bins : list
        list of numpy 1d array, bins[d] = bins of the d-th dimension


    Returns
    --------
    index (int) corresponding to the position of x in the partition
    defined by the bins.
    """
    dim = len(bins)
    flat_index = 0
    aux = 1
    assert dim == len(x_vec), "dimension mismatch in binary_search_nd()"
    for dd in range(dim):
        index_dd = np.searchsorted(bins[dd], x_vec[dd], side="right") - 1
        assert np.all(index_dd != -1), "error in binary_search_nd()"
        flat_index += aux * index_dd
        aux *= len(bins[dd]) - 1
    return flat_index


def unravel_index_uniform_bin(flat_index, dim, n_per_dim):
    index = []
    aux_index = flat_index
    for _ in range(dim):
        index.append(aux_index % n_per_dim)
        aux_index = aux_index // n_per_dim
    return tuple(index)


class DiscretizeStateWrapper(gym.ObservationWrapper):
    """
    Discretize an environment with continuous states and discrete actions.
    """

    def __init__(self, env, n_bins):
        # initialize base class
        super().__init__(env)

        self.n_bins = n_bins
        # initialize bins
        assert n_bins > 0, "DiscretizeStateWrapper requires n_bins > 0"
        n_states = 1
        tol = 1e-8
        self.dim = len(self.env.observation_space.low)
        n_states = n_bins ** self.dim
        self._bins = []
        self._open_bins = []
        for dd in range(self.dim):
            range_dd = (
                self.env.observation_space.high[dd] - self.env.observation_space.low[dd]
            )
            epsilon = range_dd / n_bins
            bins_dd = []
            for bb in range(n_bins + 1):
                val = self.env.observation_space.low[dd] + epsilon * bb
                bins_dd.append(val)
            self._open_bins.append(tuple(bins_dd[1:]))
            bins_dd[-1] += tol  # "close" the last interval
            self._bins.append(tuple(bins_dd))

            # set observation space
        self.observation_space = gym.spaces.Discrete(n_states)

        # List of discretized states
        self.discretized_states = np.zeros((self.dim, n_states))
        for ii in range(n_states):
            self.discretized_states[:, ii] = self.get_continuous_state(ii, False)

    def reset(self, **kwargs):
        tmp_state, _ = self.env.reset()
        return self.get_discrete_state(tmp_state), {}

    def step(self, action):
        next_state, reward, term, trunc, info = self.env.step(action)
        next_state = binary_search_nd(next_state, self._bins)
        return next_state, reward, term, trunc, info

    def sample(self, discrete_state, action):
        # map disctete state to continuous one
        assert self.observation_space.contains(discrete_state)
        continuous_state = self.get_continuous_state(discrete_state, randomize=True)
        # sample in the true environment
        next_state, reward, term, trunc, info = self.env.sample(
            continuous_state, action
        )
        # discretize next state
        next_state = binary_search_nd(next_state, self._bins)

        return next_state, reward, term, trunc, info

    def get_discrete_state(self, continuous_state):
        return binary_search_nd(continuous_state, self._bins)

    def get_continuous_state(self, discrete_state, randomize=False):
        assert (
            discrete_state >= 0 and discrete_state < self.observation_space.n
        ), "invalid discrete_state"
        # get multi-index
        index = unravel_index_uniform_bin(discrete_state, self.dim, self.n_bins)

        # get state
        continuous_state = np.zeros(self.dim)
        for dd in range(self.dim):
            continuous_state[dd] = self._bins[dd][index[dd]]
            if randomize:
                range_dd = (
                    self.env.observation_space.high[dd]
                    - self.env.observation_space.low[dd]
                )
                epsilon = range_dd / self.n_bins
                continuous_state[dd] += epsilon * self.rng.uniform()
        return continuous_state



def get_discrete_mountain_car_env():
    env_with_continuous_states = gym.make("MountainCar-v0", render_mode="rgb_array")
    
    env = DiscretizeStateWrapper(env_with_continuous_states, n_bins=10)
    return env
