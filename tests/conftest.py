from pathlib import Path

import pytest

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.utils.logger import Logger

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
CONFIG_FILES = [p for p in sorted(CONFIGS_DIR.glob("*.json")) if p.is_file()]


@pytest.fixture(autouse=True)
def logger():
    with Logger() as log:
        yield log


@pytest.fixture(params=CONFIG_FILES, ids=[p.name for p in CONFIG_FILES])
def train_config_path(request) -> Path:
    """Liefert den Pfad zu jeder Config-Datei direkt im configs/-Ordner."""
    return request.param


@pytest.fixture
def train_config(train_config_path: Path) -> TrainConfig:
    """Lädt und validiert die Config-Datei als vollständiges TrainConfig."""
    return TrainConfig.model_validate_json(train_config_path.read_bytes())
