"""rlcore: сид-контроль и логгер (детерминизм rollout со случайными действиями — тест главы 00, пишет автор)."""

import json

import numpy as np
import torch
from omegaconf import OmegaConf

from rlcore.config.seed import env_seeds, seed_everything
from rlcore.logging import RunLogger


def test_seed_everything_reproduces_torch_and_numpy():
    seed_everything(123)
    a = (torch.randn(3), np.random.rand(3))  # noqa: NPY002
    seed_everything(123)
    b = (torch.randn(3), np.random.rand(3))  # noqa: NPY002
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_env_seeds_are_distinct():
    assert env_seeds(10, 4) == [10, 11, 12, 13]


def test_logger_writes_files(tmp_path):
    cfg = OmegaConf.create({"seed": 5, "ppo": {"lr": 1e-3}, "env": {"name": "CartPole-v1"}})
    lg = RunLogger(tmp_path / "run", cfg, seed=5)
    lg.log_scalars({"x": 1.0}, 0)
    lg.log_hparams({"return": 1.0})
    lg.close()
    assert (tmp_path / "run" / "config.yaml").exists()
    meta = json.loads((tmp_path / "run" / "meta.json").read_text())
    assert meta["seed"] == 5 and "git_sha" in meta
    assert any((tmp_path / "run" / "tb").glob("events.out.tfevents.*"))
