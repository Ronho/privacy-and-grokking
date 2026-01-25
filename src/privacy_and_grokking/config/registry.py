from .entries import get_configs
from .model import TrainConfig


class TrainingRegistry:
    _registry: dict[str, TrainConfig] = {}
    _defaults_loaded: bool = False

    @classmethod
    def get(cls, name: str) -> TrainConfig:
        return cls._registry[name]

    @classmethod
    def register(cls, config: TrainConfig) -> None:
        if config.name in cls._registry:
            raise ValueError(
                f"Model configuration with name '{config.name}' is already registered."
            )
        cls._registry[config.name] = config

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def load_defaults(cls) -> None:
        if not cls._defaults_loaded:
            configs = get_configs()
            for config in configs:
                cls.register(config)
            cls._defaults_loaded = True
