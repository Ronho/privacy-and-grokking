import json
import logging
import logging.handlers
import sys

from datetime import datetime
from pathlib import Path
from typing import ClassVar, Literal


class StructuredFileFormatter(logging.Formatter):

    REMOVAL_KEYS: ClassVar[list[str]] = list(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())

    def __init__(self, **kwargs):
        super().__init__()
        self.default_parameters = kwargs

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            **self.default_parameters
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in StructuredFileFormatter.REMOVAL_KEYS:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def setup_logger(
    name: str,
    log_file: Path | None = None,
    log_level: str = "DEBUG",
    channel: Literal["file", "console", "all"] = "all",
    **kwargs,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if channel in ("file", "all"):
        if log_file is None:
            raise ValueError("log_file must be provided when channel is 'file' or 'all'.")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            encoding="utf-8"
        )
        file_handler.setFormatter(StructuredFileFormatter(**kwargs))
        logger.addHandler(file_handler)

    if channel in ("console", "all"):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ConsoleFormatter())
        logger.addHandler(console_handler)
    
    return logger
