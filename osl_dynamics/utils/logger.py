"""Logging utilities."""

import logging
import os
import sys
from contextlib import contextmanager


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
            for handler in logging.getLogger().handlers
            if type(handler) is logging.StreamHandler
        ]
        return self

    def __exit__(self, *args) -> None:
        for handler, stream in self._old_streams:
            handler.setStream(stream)
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        self._log_file.close()
