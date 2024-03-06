from .gridworld import GridWorld
from .chain import Chain
from .discrete_mountain_car import get_discrete_mountain_car_env

from gymnasium.envs.registration import register

register(
    id="rlberry_scool/chain",
    entry_point="rlberry_scool.envs.finite:Chain",
    max_episode_steps=1000,
)
