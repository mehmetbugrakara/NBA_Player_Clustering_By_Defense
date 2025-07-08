import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler


# Create singleton logger instance
logger = logging.getLogger("NBA Player Cluster By Defensive Shot Base")
logger.setLevel(logging.DEBUG)

# Avoid duplicate handlers
if not logger.handlers:
    file_path = "logs/analytic.log"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    rotating_handler = TimedRotatingFileHandler(filename=file_path, when='D', interval=30, backupCount=6)
    formatter = logging.Formatter(fmt="%(levelname)s %(asctime)s %(message)s")
    rotating_handler.setFormatter(formatter)
    rotating_handler.setLevel(logging.INFO)

    class DebugStreamHandler(logging.StreamHandler):
        def emit(self, record):
            if record.levelno == logging.DEBUG:
                super().emit(record)

    stream_handler = DebugStreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)

    logger.addHandler(rotating_handler)
    logger.addHandler(stream_handler)


# Correct metaclass usage
class _Base(type):
    """
    Metaclass for injecting logger into instances
    """
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.logger = logger
        return instance
