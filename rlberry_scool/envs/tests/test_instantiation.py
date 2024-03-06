import numpy as np
import pytest

from rlberry.envs import gym_make, PipelineEnv
from rlberry_scool.envs.finite import Chain
from rlberry_scool.envs.finite import GridWorld
from rlberry_scool.envs.finite import get_discrete_mountain_car_env
from rlberry.rendering.render_interface import RenderInterface2D

classes = [
    GridWorld,
    Chain,
]


@pytest.mark.parametrize("ModelClass", classes)
def test_instantiation(ModelClass):
    env = ModelClass()

    if env.is_online():
        for _ in range(2):
            state, info = env.reset()
            for _ in range(50):
                assert env.observation_space.contains(state)
                action = env.action_space.sample()
                next_s, _, _, _, _ = env.step(action)
                state = next_s

    if env.is_generative():
        for _ in range(100):
            state = env.observation_space.sample()
            action = env.action_space.sample()
            next_s, _, _, _, _ = env.sample(state, action)
            assert env.observation_space.contains(next_s)


@pytest.mark.parametrize("ModelClass", classes)
def test_rendering_calls(ModelClass):
    env = ModelClass()
    if isinstance(env, RenderInterface2D):
        _ = env.get_background()
        _ = env.get_scene(env.observation_space.sample())


def test_gridworld_aux_functions():
    env = GridWorld(
        nrows=5, ncols=8, walls=((1, 1),), reward_at={(4, 4): 1, (4, 3): -1}
    )
    env.log()  # from FiniteMDP
    env.render_ascii()  # from GridWorld
    vals = np.arange(env.observation_space.n)
    env.display_values(vals)
    env.print_transition_at(0, 0, "up")

    layout = env.get_layout_array(vals, fill_walls_with=np.inf)
    for rr in range(env.nrows):
        for cc in range(env.ncols):
            if (rr, cc) in env.walls:
                assert layout[rr, cc] == np.inf
            else:
                assert layout[rr, cc] == vals[env.coord2index[(rr, cc)]]


def test_gridworld_from_layout():
    layout = """
    IOOOO # OOOOO  O OOOOR
    OOOOO # OOOOO  # OOOOO
    OOOOO O OOOOO  # OOTOO
    OOOOO # OOOOO  # OOOOO
    IOOOO # OOOOO  # OOOOr"""
    env = GridWorld.from_layout(layout)
    env.reset()


def test_pipeline():
    from rlberry.wrappers import RescaleRewardWrapper
    from rlberry.wrappers.discretize_state import DiscretizeStateWrapper

    env_ctor, env_kwargs = PipelineEnv, {
        "env_ctor": gym_make,
        "env_kwargs": {"id": "Acrobot-v1"},
        "wrappers": [(RescaleRewardWrapper, {"reward_range": (0, 1)})],
    }
    env = env_ctor(**env_kwargs)
    _, reward, _, _, _ = env.step(0)
    assert (reward <= 1) and (reward >= 0)

    env_ctor, env_kwargs = PipelineEnv, {
        "env_ctor": gym_make,
        "env_kwargs": {"id": "Acrobot-v1"},
        "wrappers": [
            (RescaleRewardWrapper, {"reward_range": (0, 1)}),
            (DiscretizeStateWrapper, {"n_bins": 10}),
        ],
    }
    env = env_ctor(**env_kwargs)
    # check that wrapped in the right order
    assert isinstance(
        env.env, RescaleRewardWrapper
    ), "the environments in Pipeline env may not be wrapped in order"
    assert isinstance(env.env.env, DiscretizeStateWrapper)


def test_discrete_mc():
    env = get_discrete_mountain_car_env()
    env.reset()
    done = False
    while not done:
        a = env.action_space.sample()
        _, _, term, trunc, _ = env.step(a)
        done = term or trunc
