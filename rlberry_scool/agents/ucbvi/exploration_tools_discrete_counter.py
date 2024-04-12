from abc import ABC
import numpy as np

from rlberry.spaces import Discrete
from rlberry.utils.space_discretizer import Discretizer


class DiscreteCounter(ABC):
    """
    Parameters
    ----------
    observation_space : spaces.Box or spaces.Discrete
    action_space : spaces.Box or spaces.Discrete
    n_bins_obs: int
        number of bins to discretize observation space
    n_bins_actions: int
        number of bins to discretize action space
    rate_power : float
        Returns bonuses in 1/n ** rate_power.
    **kwargs : Keyword Arguments
        Extra arguments. Not used here.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        n_bins_obs=10,
        n_bins_actions=10,
        rate_power=0.5,
        **kwargs
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space

        self.rate_power = rate_power

        self.continuous_state = False
        self.continuous_action = False

        if isinstance(observation_space, Discrete):
            self.n_states = observation_space.n
        else:
            self.continuous_state = True
            self.state_discretizer = Discretizer(self.observation_space, n_bins_obs)
            self.n_states = self.state_discretizer.discrete_space.n

        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
        else:
            self.continuous_action = True
            self.action_discretizer = Discretizer(self.action_space, n_bins_actions)
            self.n_actions = self.action_discretizer.discrete_space.n

        self.N_sa = np.zeros((self.n_states, self.n_actions))

    def _preprocess(self, state, action):
        if self.continuous_state:
            state = self.state_discretizer.discretize(state)
        if self.continuous_action:
            action = self.action_discretizer.discretize(action)
        return state, action

    def reset(self):
        self.N_sa = np.zeros((self.n_states, self.n_actions))


    def update(self, state, action, next_state=None, reward=None, **kwargs):
        """
        **kwargs : Keyword Arguments
            Extra arguments. Not used here.
        """
        state, action = self._preprocess(state, action)
        self.N_sa[state, action] += 1

    def get_n_visited_states(self):
        """
        Returns the number of different states sent to the .update() function.
        For continuous state spaces, counts the number of different discretized states.
        """
        n_visited_states = (self.N_sa.sum(axis=1) > 0).sum()
        return n_visited_states
