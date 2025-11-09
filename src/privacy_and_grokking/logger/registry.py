import logging

from .formatter import setup_logger

_LOGGER_REGISTRY: dict[str, logging.Logger] = {}
_DEFAULT_LOGGER_NAME = "default"
_LOGGER_REGISTRY[_DEFAULT_LOGGER_NAME] = setup_logger(_DEFAULT_LOGGER_NAME, log_level="DEBUG", channel="console")

def register_logger(name: str, overwrite: bool = False, **kwargs) -> logging.Logger:
    if name in _LOGGER_REGISTRY and not overwrite:
        raise ValueError(f"Logger '{name}' is already registered.")
    _LOGGER_REGISTRY[name] = setup_logger(name, **kwargs)
    return _LOGGER_REGISTRY[name]

def get_logger(name: str | None = None) -> logging.Logger:
    if name is None:
        name = _DEFAULT_LOGGER_NAME
    if name not in _LOGGER_REGISTRY:
        raise ValueError(f"Logger '{name}' not registered. Call register_logger first.")
    return _LOGGER_REGISTRY[name]
