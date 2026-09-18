"""Сид-контроль.

Один ``seed`` из конфига раздаётся всем источникам случайности:
    random / numpy / torch (CPU и все CUDA-устройства) — здесь;
    среды — через ``env_seeds``: у каждой из N параллельных сред свой сид seed+i,
    иначе все копии среды идут по одной траектории и векторизация бесполезна.
``deterministic=True`` включает детерминированные cuDNN/torch-алгоритмы (медленнее); нужно только
для тестов равенства двух запусков, в обучении обычно выключено.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 — сидим именно глобальный legacy-RNG: его используют сторонние библиотеки
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def env_seeds(seed: int, num_envs: int) -> list[int]:
    return [seed + i for i in range(num_envs)]
