"""Logging utilities."""

import logging
import os
import sys
from contextlib import contextmanager
from typing import Union

_logger = logging.getLogger("osl-dynamics")


class _Formatter(logging.Formatter):
    """Formats messages as :code:`[HH:MM:SS] message`.

    Warnings and errors are prefixed with their level. In debug mode, the
    file, line and function a message came from are also shown.
    """

    def __init__(self, debug: bool = False):
        if debug:
            fmt = "[%(asctime)s %(filename)s:%(lineno)d:%(funcName)s] %(message)s"
        else:
            fmt = "[%(asctime)s] %(message)s"
        super().__init__(fmt, datefmt="%H:%M:%S")

    def formatMessage(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.WARNING:
            record.message = f"{record.levelname}: {record.message}"
        return super().formatMessage(record)


# Handler that prints the messages of osl-dynamics
_handler = logging.StreamHandler()


def _setup_logger() -> None:
    """Setup the osl-dynamics logger.

    Messages are printed by a handler of the osl-dynamics logger (not the
    root logger), so the logging of other packages is unaffected.
    """
    if _handler not in _logger.handlers:
        _logger.addHandler(_handler)
    _logger.propagate = False
    level = os.environ.get("OSL_DYNAMICS_VERBOSITY", "INFO")
    try:
        set_verbosity(level)
    except ValueError:
        set_verbosity("INFO")
        _logger.warning(f"Invalid OSL_DYNAMICS_VERBOSITY: {level}. Using INFO.")


def set_verbosity(level: Union[str, int]) -> None:
    """Set how much osl-dynamics prints.

    Parameters
    ----------
    level : str or int
        Logging level. One of:

        - :code:`'WARNING'`: only print warnings and errors.
        - :code:`'INFO'`: also print progress messages and progress bars
          (the default).
        - :code:`'DEBUG'`: also print debugging messages and show where each
          message came from.

        The level can also be set with the :code:`OSL_DYNAMICS_VERBOSITY`
        environment variable.
    """
    if isinstance(level, str):
        level = level.upper()
    _logger.setLevel(level)
    _handler.setFormatter(_Formatter(debug=_logger.isEnabledFor(logging.DEBUG)))


@contextmanager
def set_logging_level(logger: logging.Logger, level: int) -> None:
    """Temporarily change the logging level of a logger.

    Parameters
    ----------
    logger : logging.Logger
        Logger to change.
    level : int
        Logging level to set (e.g. ``logging.WARNING``).
    """
    current_level = logger.getEffectiveLevel()
    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(current_level)


class MEEGSessionLogger:
    """Redirects all stdout/stderr to a per-session log file.

    Progress messages can be printed to screen using the ``log`` method.

    Parameters
    ----------
    session : str
        Session identifier used as a prefix in log messages.
    log_dir : str
        Directory to write log files to.

    Usage
    -----
    with MEEGSessionLogger("sub-01_task-rest", log_dir="logs") as logger:
        logger.log("Filtering...")
        raw.resample(250)  # verbose output goes to log file only
        logger.log("Done.")

    Screen output:
        [sub-01_task-rest] Filtering...
        [sub-01_task-rest] Done.
    """

    def __init__(self, session: str, log_dir: str):
        self.session = session
        self.prefix = f"[{session}] "
        self.log_dir = log_dir
        self.warnings = []

    def _timestamp(self) -> str:
        """Return current time as HH:MM:SS string."""
        from datetime import datetime

        return datetime.now().strftime("%H:%M:%S")

    def _write_line(self, stream, msg: str) -> None:
        """Write a line to a stream (the screen) and the log file."""
        line = f"[{self._timestamp()} {self.session}] {msg}\n"
        stream.write(line)
        stream.flush()
        self._log_file.write(line)
        self._log_file.flush()

    def log(self, msg: str) -> None:
        """Print a progress message to screen (and log file)."""
        self._write_line(sys.__stdout__, msg)

    def warning(self, msg: str) -> None:
        """Print a warning message to screen (and log file).

        Warnings are kept in ``self.warnings``. They are for a session that
        finishes without everything it was meant to produce.
        """
        self.warnings.append(msg)
        self._write_line(sys.__stderr__, f"WARNING: {msg}")

    def error(self, msg: str) -> None:
        """Print an error message to screen (and log file)."""
        self._write_line(sys.__stderr__, f"ERROR: {msg}")

    def write(self, text: str) -> None:
        if text:
            self._log_file.write(text)
            self._log_file.flush()

    def flush(self) -> None:
        self._log_file.flush()

    def __enter__(self) -> "MEEGSessionLogger":
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"{self.session}.log")
        self._log_file = open(log_path, "w")
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

        # Handlers of the logging module hold the stream they were created
        # with, so they are pointed at the log file too
        self._old_streams = [
            (handler, handler.setStream(self))
            for logger in [logging.getLogger(), _logger]
            for handler in logger.handlers
            if type(handler) is logging.StreamHandler
        ]
        return self

    def __exit__(self, *args) -> None:
        for handler, stream in self._old_streams:
            handler.setStream(stream)
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        self._log_file.close()
