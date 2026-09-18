"""TensorBoard-логгер прогона.

Один прогон = одна папка ``log_dir`` (``runs/<chapter>/<run_name>/``):
    tb/            event-файлы TensorBoard (scalars, hparams, text)
    config.yaml    полностью разрешённый конфиг Hydra
    meta.json      git-хеш (+dirty), hostname, время старта, версии, сид
    hydra/         снимок Hydra (config/overrides), кладёт сама Hydra
    log.txt        stdout запуска (пишет scripts/run.sh)
Никаких внешних сервисов: смотреть через ``tensorboard --logdir runs/<chapter>`` и ssh-туннель.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter


def git_info(repo: Path | None = None) -> dict[str, str | bool]:
    """Хеш HEAD и флаг незакоммиченных изменений; без git — 'unknown'."""
    try:
        cwd = str(repo) if repo else None
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, text=True).strip())
        return {"git_sha": sha, "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_sha": "unknown", "git_dirty": False}


def run_metadata(seed: int) -> dict[str, Any]:
    return {
        **git_info(),
        "hostname": socket.gethostname(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": seed,
        "argv": sys.argv,
    }


def _flatten(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v if isinstance(v, (int, float, str, bool)) else str(v)
    return out


class RunLogger:
    """Обёртка над SummaryWriter: скаляры по шагу, hparams в конце, конфиг и meta на диске."""

    def __init__(self, log_dir: str | Path, cfg: DictConfig, seed: int):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.dir / "tb"))
        self.cfg = OmegaConf.to_container(cfg, resolve=True)
        self.meta = run_metadata(seed)
        (self.dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
        (self.dir / "meta.json").write_text(
            json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # конфиг и meta видны во вкладке Text; git-хеш — первое, что нужно при разборе старого прогона
        self.writer.add_text("run/config", "```yaml\n" + OmegaConf.to_yaml(cfg, resolve=True) + "\n```", 0)
        self.writer.add_text("run/meta", "```json\n" + json.dumps(self.meta, indent=2) + "\n```", 0)

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        for k, v in scalars.items():
            self.writer.add_scalar(k, float(v), step)

    def log_hparams(self, metrics: dict[str, float]) -> None:
        """Вкладка HParams: плоский конфиг + git-хеш против финальных метрик. Вызывать один раз в конце."""
        hp = _flatten(self.cfg)  # type: ignore[arg-type]
        hp["git_sha"] = str(self.meta["git_sha"])
        self.writer.add_hparams(hp, {f"final/{k}": float(v) for k, v in metrics.items()}, run_name=".")

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
