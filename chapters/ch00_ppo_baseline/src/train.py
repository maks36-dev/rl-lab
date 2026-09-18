"""Точка входа главы 00. Пока: конфиг → сиды → среда → логгер → выйти (T0.2–T0.4). Обучение — T2.x.

Запуск (из корня репо, после `source scripts/env.sh`):
    python -m chapters.ch00_ppo_baseline.src.train env=cartpole seed=1
    python -m chapters.ch00_ppo_baseline.src.train --multirun seed=1,2,3
    scripts/run.sh ch00_ppo_baseline <run_name> env=halfcheetah seed=1   # tmux + свободная GPU
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from rlcore.config import TrainCfg, register
from rlcore.config.seed import seed_everything
from rlcore.envs import make_vec_env
from rlcore.logging import RunLogger

register()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    typed: TrainCfg = OmegaConf.to_object(cfg)  # type: ignore[assignment]
    seed_everything(typed.seed)
    logger = RunLogger(typed.log_dir, cfg, typed.seed)
    envs = make_vec_env(typed.env, typed.seed)
    obs, _ = envs.reset(seed=typed.seed)
    print(
        f"[train] {typed.run_name}: env={typed.env.name} x{typed.env.num_envs} obs{obs.shape} "
        f"device={typed.device} git={logger.meta['git_sha']} log_dir={logger.dir}"
    )
    logger.log_scalars({"debug/obs_mean": float(obs.mean())}, step=0)
    logger.log_hparams({"return": 0.0})
    logger.close()
    envs.close()


if __name__ == "__main__":
    main()
