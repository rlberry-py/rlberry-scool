from rlberry_scool.envs.finite import Chain, GridWorld
from rlberry_scool.agents.dynprog import ValueIterationAgent
import tempfile
import os


def test_chain_rendering():
    env = Chain(10, 0.3)
    env.enable_rendering()
    for tt in range(5):
        env.step(env.action_space.sample())
    with tempfile.TemporaryDirectory() as tmpdirname:
        saving_path = tmpdirname + "/test_gif.gif"
        env.save_gif(saving_path)
        assert os.path.isfile(saving_path)
        try:
            os.remove(saving_path)
        except Exception:
            pass


def test_gridworld_rendering():
    env = GridWorld(7, 10, walls=((2, 2), (3, 3)))

    agent = ValueIterationAgent(env, gamma=0.95)
    info = agent.fit()
    print(info)

    env.enable_rendering()
    observation, info = env.reset()
    for tt in range(50):
        action = agent.policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            # Warning: this will never happen in the present case because there is no terminal state.
            # See the doc of GridWorld for more informations on the default parameters of GridWorld.
            break

    with tempfile.TemporaryDirectory() as tmpdirname:
        saving_path = tmpdirname + "/test_gif.gif"
        env.save_gif(saving_path)
        assert os.path.isfile(saving_path)
        try:
            os.remove(saving_path)
        except Exception:
            pass
