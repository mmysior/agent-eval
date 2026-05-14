import logging

import colorlog
from tqdm import tqdm


class TqdmHandler(colorlog.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))


def setup_logging():
    root_logger = logging.getLogger()

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
