"""Композиция конфига главы 00: схема, группа env, переопределения, интерполяции."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from rlcore.config import TrainCfg, register

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(autouse=True)
def _hydra():
    GlobalHydra.instance().clear()
    register()
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        yield


def test_default_is_cartpole_and_typed():
    cfg = compose(config_name="config")
    typed: TrainCfg = OmegaConf.to_object(cfg)
    assert typed.env.name == "CartPole-v1" and typed.env.continuous is False
    assert typed.run_name == "CartPole-v1_s1"
    assert typed.log_dir.endswith("CartPole-v1_s1")


def test_env_override_and_seed_interpolation():
    cfg = compose(config_name="config", overrides=["env=halfcheetah", "seed=7"])
    typed: TrainCfg = OmegaConf.to_object(cfg)
    assert typed.env.name == "HalfCheetah-v5" and typed.env.continuous is True
    assert typed.run_name == "HalfCheetah-v5_s7"


def test_unknown_field_is_rejected():
    with pytest.raises(ConfigCompositionException):
        compose(config_name="config", overrides=["total_stpes=5"])


def test_wrong_type_is_rejected():
    with pytest.raises(ConfigCompositionException):
        compose(config_name="config", overrides=["ppo.lr=fast"])
