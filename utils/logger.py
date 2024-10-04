import logging
import os
import time
from typing import Optional


class LoggerSingleton:
    _instance: Optional[logging.Logger] = None
    _initialized: bool = False

    @classmethod
    def get_logger(cls) -> logging.Logger:
        if cls._instance is None:
            cls._instance = logging.getLogger(__name__)
            cls._instance.setLevel(logging.INFO)
        return cls._instance

    @classmethod
    def configure(cls, log_path: str, log_level=logging.INFO, suffix: str = ""):
        if not cls._initialized:
            logger = cls.get_logger()
            logger.handlers.clear()  # Clear any existing handlers

            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            cur_time = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime(time.time()))
            file_handler = logging.FileHandler(
                os.path.join(log_path, f"{suffix}_{cur_time}.log")
            )
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # console_handler = logging.StreamHandler()
            # console_handler.setLevel(logging.INFO)
            # console_handler.setFormatter(formatter)
            # logger.addHandler(console_handler)

            cls._initialized = True


def get_logger() -> logging.Logger:
    return LoggerSingleton.get_logger()


def configure_logger(config, suffix: str = ""):
    LoggerSingleton.configure(config.logging.log_path, suffix)
