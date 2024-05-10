import logging
import sys
import os

MIN_LEVEL = logging.DEBUG
MESSAGE = 25
logging.addLevelName(MESSAGE, "MESSAGE")
LOGGING_LEVEL = 25


class LogFilter(logging.Filter):
    """Filters (lets through) all messages with level < LEVEL"""
    # http://stackoverflow.com/a/24956305/408556
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        # "<" instead of "<=": since logger.setLevel is inclusive, this should
        # be exclusive
        return record.levelno < self.level


class Logger(logging.Logger):
    def message(self, msg, *args, **kwargs):
        if self.isEnabledFor(MESSAGE):
            self._log(MESSAGE, msg, args, **kwargs)


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        use_color = self.supports_color()
        if record.levelno == logging.WARNING:
            if use_color:
                record.msg = f"\033[93m{record.msg}\033[0m"
        elif record.levelno == logging.ERROR:
            if use_color:
                record.msg = f"\033[91m{record.msg}\033[0m"
        elif record.levelno == 25 or record.levelno == logging.INFO:
            if use_color:
                record.msg = f"\033[96m{record.msg}\033[0m"
        return super().format(record)

    def supports_color(self):
        """Check if the system supports ANSI color formatting.
        """
        supported_platform = 'ANSICON' in os.environ
        is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        return supported_platform and is_a_tty


def get_logger(module_name, level=LOGGING_LEVEL):
    logging.setLoggerClass(Logger)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stdout_handler.addFilter(LogFilter(logging.WARNING))
    stdout_handler.setLevel(level)
    if level == 25:
        formatter = ColoredFormatter("[%(levelname)s]: %(message)s")
    else:
        formatter = ColoredFormatter(
            "[%(levelname)s] %(name)s.%(funcName)s (line %(lineno)d): %(message)s"
        )
    stdout_handler.setFormatter(formatter)
    stderr_handler.setLevel(max(MIN_LEVEL, logging.WARNING))
    # messages lower than WARNING go to stdout
    # messages >= WARNING (and >= STDOUT_LOG_LEVEL) go to stderr
    logger = logging.getLogger(module_name)
    logger.propagate = False
    logger.handlers = [stdout_handler, stderr_handler]
    logger.setLevel(level)
    return logger
