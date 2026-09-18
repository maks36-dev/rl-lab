"""Точка входа главы 00. Пока: собрать конфиг, показать его и выйти (T0.2).

Запуск (из корня репо, после `source scripts/env.sh`):
    python -m chapters.ch00_ppo_baseline.src.train env=cartpole seed=1
    python -m chapters.ch00_ppo_baseline.src.train --multirun seed=1,2,3
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from rlcore.config import TrainCfg, register

register()


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # OmegaConf.to_object проверяет, что конфиг совпадает со схемой (типы, отсутствие MISSING)
    typed: TrainCfg = OmegaConf.to_object(cfg)  # type: ignore[assignment]
    print(OmegaConf.to_yaml(cfg))
    print(f"[train] run_name={typed.run_name} device={typed.device} env={typed.env.name}")


if __name__ == "__main__":
    main()
