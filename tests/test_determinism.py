import numpy as np
import numpy.typing as npt

from rlcore.config import EnvCfg
from rlcore.config.seed import seed_everything
from rlcore.envs import make_vec_env


def collect_obs(seed: int) -> npt.NDArray[np.float32]:
    seed_everything(seed)
    envs = make_vec_env(EnvCfg(name="CartPole-v1", num_envs=4), seed)
    envs.action_space.seed(seed)

    trajectory = []
    for _ in range(100):
        actions = envs.action_space.sample()
        obs, *_ = envs.step(actions)
        trajectory.append(obs)

    return np.array(trajectory)


def test_same_seed_same_obs():
    a = collect_obs(1)
    b = collect_obs(1)
    assert np.array_equal(a, b)


def test_different_seed_different_obs():
    a = collect_obs(1)
    b = collect_obs(2)
    assert not np.array_equal(a, b)
