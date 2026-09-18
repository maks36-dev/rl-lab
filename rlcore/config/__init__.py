"""Схема конфигов курса.

Hydra собирает YAML-файлы в один DictConfig; эти dataclass'ы — его типизированный скелет.
Регистрируем схему в ConfigStore под именем ``base_schema``; YAML главы наследует её через ``defaults``.
Что даёт: опечатка в имени поля (``total_stpes=1``) или неверный тип падают на старте, а не через час обучения.
"""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class EnvCfg:
    name: str = MISSING  # id среды в gymnasium, например CartPole-v1
    num_envs: int = 1  # параллельных копий среды (векторизация)
    continuous: bool = False  # непрерывные действия → гауссова политика
    max_episode_steps: int | None = None  # None = дефолт среды


@dataclass
class PPOCfg:
    """Гиперпараметры PPO. Значения по умолчанию — из CleanRL; пересматриваются в T1.2 (37 деталей)."""

    lr: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    rollout_steps: int = 128  # шагов на каждую среду до обновления
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    target_kl: float | None = None  # ранняя остановка эпох по approx_kl


@dataclass
class TrainCfg:
    env: EnvCfg = field(default_factory=EnvCfg)
    ppo: PPOCfg = field(default_factory=PPOCfg)
    seed: int = 1
    device: str = "cuda"
    total_steps: int = 100_000  # шагов среды всего (по всем num_envs)
    eval_every: int = 10_000
    run_name: str = MISSING  # задаётся в YAML через интерполяцию
    log_dir: str = MISSING  # куда TensorBoard и чекпоинты


def register() -> None:
    """Регистрирует схему; вызывается до ``@hydra.main``."""
    cs = ConfigStore.instance()
    cs.store(name="base_schema", node=TrainCfg)
