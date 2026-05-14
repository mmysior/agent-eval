import logging

import colorlog
from langfuse import Langfuse
from tqdm import tqdm

from agent_eval.core.config import config


class TqdmHandler(colorlog.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))


def setup_logging():
    root_logger = logging.getLogger()

    # Don't add handlers if they already exist (prevents duplicate logs)
    if root_logger.handlers:
        return

    handler = TqdmHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s:%(reset)s %(asctime)s [%(name)s] %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


def setup_langfuse():
    if config.LANGFUSE_TRACING_ENABLED:
        Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_BASE_URL,
        )
