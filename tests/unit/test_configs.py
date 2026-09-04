from pathlib import Path

from privacy_and_grokking.config import TrainConfig


def test_config_validates(train_config_path: Path):
    """Each JSON config file must parse into a valid TrainConfig."""
    cfg = TrainConfig.model_validate_json(train_config_path.read_bytes())
    assert isinstance(cfg, TrainConfig)
    assert cfg.model is not None
    assert cfg.data is not None
    assert cfg.loss is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None


def test_config_roundtrips(train_config: TrainConfig):
    """Serializing and re-parsing a config must produce an identical object."""
    roundtripped = TrainConfig.model_validate(train_config.model_dump())
    assert train_config == roundtripped


def test_config_name_property(train_config: TrainConfig):
    """The name property must return a non-empty string."""
    assert isinstance(train_config.name, str)
    assert len(train_config.name) > 0
    assert isinstance(train_config.full_name, str)
    assert len(train_config.full_name) > 0
