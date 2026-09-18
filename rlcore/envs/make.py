"""Фабрика векторизованной среды gymnasium по ``EnvCfg``.

SyncVectorEnv: N копий среды в одном процессе, шаг возвращает батч (N, obs_dim).
RecordEpisodeStatistics кладёт return/length эпизода в ``info`` — источник метрик episode/*.
Сид: ``reset(seed=[...])`` с разным сидом на копию (см. rlcore.config.seed.env_seeds).
"""

from __future__ import annotations

import gymnasium as gym

from rlcore.config import EnvCfg
from rlcore.config.seed import env_seeds


def make_vec_env(cfg: EnvCfg, seed: int) -> gym.vector.SyncVectorEnv:
    def thunk():
        env = gym.make(cfg.name, max_episode_steps=cfg.max_episode_steps)
        return gym.wrappers.RecordEpisodeStatistics(env)

    envs = gym.vector.SyncVectorEnv([thunk for _ in range(cfg.num_envs)])
    envs.reset(seed=env_seeds(seed, cfg.num_envs))
    return envs
