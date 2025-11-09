from typing import Callable

class TrainingRegistry:
    _registry: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, func: Callable):
        cls._registry[name] = func

    @classmethod
    def get(cls, name: str) -> Callable:
        return cls._registry[name]

    @classmethod
    def list_models(cls) -> list[str]:
        return list(cls._registry.keys())
