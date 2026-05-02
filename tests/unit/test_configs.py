"""Tests that all JSON config files in configs/ validate against TrainConfig."""

from pathlib import Path

import pytest

from privacy_and_grokking.config import TrainConfig

CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"
CONFIG_FILES = sorted(CONFIGS_DIR.glob("*.json"))


@pytest.mark.parametrize(
    "config_path",
    CONFIG_FILES,
    ids=[f.stem for f in CONFIG_FILES],
)
def test_config_validates(config_path: Path):
    """Each JSON config file must parse into a valid TrainConfig."""
    cfg = TrainConfig.model_validate_json(config_path.read_bytes())
    assert cfg.model is not None
    assert cfg.data is not None
    assert cfg.loss is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None


@pytest.mark.parametrize(
    "config_path",
    CONFIG_FILES,
    ids=[f.stem for f in CONFIG_FILES],
)
def test_config_roundtrips(config_path: Path):
    """Serializing and re-parsing a config must produce an identical object."""
    cfg = TrainConfig.model_validate_json(config_path.read_bytes())
    roundtripped = TrainConfig.model_validate(cfg.model_dump())
    assert cfg == roundtripped


@pytest.mark.parametrize(
    "config_path",
    CONFIG_FILES,
    ids=[f.stem for f in CONFIG_FILES],
)
def test_config_name_property(config_path: Path):
    """The name property must return a non-empty string."""
    cfg = TrainConfig.model_validate_json(config_path.read_bytes())
    assert isinstance(cfg.name, str)
    assert len(cfg.name) > 0
    assert isinstance(cfg.full_name, str)
    assert len(cfg.full_name) > 0
