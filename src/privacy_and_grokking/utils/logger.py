import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
from structlog.types import FilteringBoundLogger


class Logger:
    _instance: Optional["Logger"] = None

    def __new__(cls) -> "Logger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance.log_file_path = None
            cls._instance.logger = None
            cls._instance._handlers = []
            cls._instance._temp_file = None
        return cls._instance

    def setup(self) -> FilteringBoundLogger:
        if not self._initialized:
            log_level = os.getenv("LOG_LEVEL", "INFO").upper()
            level = getattr(logging, log_level, logging.INFO)

            prefix = datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + "_"
            self._temp_file = tempfile.NamedTemporaryFile(
                suffix=".log", prefix=prefix, delete=False
            )
            self.log_file_path = Path(self._temp_file.name)
            self._temp_file.close()  # Close FD so handlers can take over

            formatter = logging.Formatter("%(message)s")
            self._handlers = [
                logging.FileHandler(self.log_file_path),
                logging.StreamHandler(sys.stdout),
            ]

            app_logger = logging.getLogger("privacy_and_grokking")
            app_logger.setLevel(level)
            app_logger.propagate = False  # don't bubble up to root; keeps external loggers out
            for handler in self._handlers:
                handler.setFormatter(formatter)
                handler.setLevel(level)
                app_logger.addHandler(handler)

            structlog.configure(
                processors=[
                    structlog.contextvars.merge_contextvars,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.add_log_level,
                    structlog.processors.JSONRenderer(),
                ],
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.make_filtering_bound_logger(level),
                cache_logger_on_first_use=True,
            )

            self.logger = structlog.get_logger()
            self._initialized = True
            self.logger.info("Logger initialized", log_file=str(self.log_file_path))

        return self.logger

    def __enter__(self) -> FilteringBoundLogger:
        return self.setup()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.cleanup()
        return False

    def cleanup(self) -> None:
        """Teardown logic to remove handlers and delete the temp file."""
        if not self._initialized:
            return

        app_logger = logging.getLogger("privacy_and_grokking")
        for handler in self._handlers:
            handler.flush()
            handler.close()
            app_logger.removeHandler(handler)
        self._handlers.clear()

        if self.log_file_path and self.log_file_path.exists():
            try:
                self.log_file_path.unlink()
            except Exception as e:
                print(f"Failed to remove log file {self.log_file_path}: {e}")

        self.log_file_path = None
        self._initialized = False
        Logger._instance = None

    @classmethod
    def get(cls) -> FilteringBoundLogger:
        if cls._instance is None or not cls._instance._initialized:
            raise RuntimeError(
                "Logger has not been initialized. "
                "Ensure you are inside a 'with Logger()' block or have called 'Logger().setup()' first."
            )
        return cls._instance.logger
