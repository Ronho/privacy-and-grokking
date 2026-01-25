import re

from pathlib import Path
from typing import Any, TypedDict
from ..logger import get_logger


logger = get_logger()

class CheckpointPaths(TypedDict):
    model: Path
    optimizer: Path
    onnx: Path

class PathKeeper:

    _CACHE = "cache"
    _RUN_FOLDER = "data/runs/{run_id}/"
    _LOG = _RUN_FOLDER + "logs/{log_id}.log"
    _IMAGE_FOLDER = _RUN_FOLDER + "images/{model}/"
    _ATTACK_FOLDER = _RUN_FOLDER + "attacks/{model}/"
    _TRAIN_FOLDER = _RUN_FOLDER + "models/{model}/"
    _CHECKPOINT = _TRAIN_FOLDER + "checkpoints/{step}/"

    def __init__(self, base_dir: Path | str | None = None, create_dirs: bool = True):
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent.parent.parent
        elif isinstance(base_dir, str):
            self.base_dir = Path(base_dir)
        
        self.required_params = {}
        for key, value in self.__class__.__dict__.items():
            if not key.startswith("__") and isinstance(value, str):
                self.required_params[value] = re.findall(r"{(.*?)}", value)
        self.all_params = set(param for sublist in self.required_params.values() for param in sublist)
        self.params = {}
        self.create_dirs = create_dirs

    def set_params(self, params: dict[str, Any]):
        for key, value in params.items():
            if key not in self.all_params:
                logger.warning(f"Parameter '{key}' is not recognized. Ignoring.")
                continue
            if value is None:
                logger.warning(f"Parameter '{key}' is set to None. Ignoring.")
                continue
            self.params[key] = value

    def _fill(self, path_template: str) -> Path:
        if not all(param in self.params for param in self.required_params[path_template]):
            missing = [param for param in self.required_params[path_template] if param not in self.params]
            logger.warning(f"Missing parameters for path template '{path_template}'.", extra={"missing_parameters": missing})

        filled = path_template.format(**self.params)
        path = self.base_dir / filled
        if not "*" in str(path) and self.create_dirs:
            if "." in path.name:
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def CACHE(self) -> Path:
        return self._fill(PathKeeper._CACHE)

    @property
    def LOG(self) -> Path:
        return self._fill(PathKeeper._LOG)
    
    @property
    def TRAIN_CONFIG(self) -> Path:
        return self._fill(PathKeeper._TRAIN_FOLDER) / "train_config.json"

    @property
    def TRAIN_METRICS(self) -> Path:
        return self._fill(PathKeeper._TRAIN_FOLDER) / "train_metrics.json"
    
    @property
    def MODEL_TORCH(self) -> Path:
        return self._fill(PathKeeper._CHECKPOINT) / "model.pt"

    @property
    def MODEL_ONNX(self) -> Path:
        return self._fill(PathKeeper._CHECKPOINT) / "model.onnx"

    @property
    def OPTIMIZER(self) -> Path:
        return self._fill(PathKeeper._CHECKPOINT) / "optimizer.pt"
    
    @property
    def RNG_STATE(self) -> Path:
        return self._fill(PathKeeper._CHECKPOINT) / "rng_state.pt"
    
    @property
    def TRAIN_LOGITS(self) -> Path:
        return self._fill(PathKeeper._CHECKPOINT) / "train_logits.parquet"
    
    @property
    def TEST_LOGITS(self) -> Path:
        return self._fill(PathKeeper._CHECKPOINT) / "test_logits.parquet"
    
    @property
    def IMAGE_FOLDER(self) -> Path:
        return self._fill(PathKeeper._IMAGE_FOLDER)
    
    @property
    def ATTACK_FOLDER(self) -> Path:
        return self._fill(PathKeeper._ATTACK_FOLDER)
