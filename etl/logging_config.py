"""Centralised logging configuration for the ETL pipeline.

Call ``setup_logging()`` once at process start (in ``main.py``).
Every other module obtains its logger with ``logging.getLogger(__name__)``,
which places it under the ``etl.*`` hierarchy and picks up these handlers
automatically without touching third-party loggers (soccerdata, selenium…).

Each process invocation writes to its own timestamped file inside ``logs/``:
    logs/etl_2026-06-04_01-33-27.log
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_COLORS: dict[int, str] = {
    logging.DEBUG:    "\033[36m",
    logging.INFO:     "\033[32m",
    logging.WARNING:  "\033[33m",
    logging.ERROR:    "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Adds ANSI colour to the level name without mutating the shared LogRecord."""

    def format(self, record: logging.LogRecord) -> str:
        record = logging.makeLogRecord(record.__dict__)
        color = _COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


def setup_logging(
    logs_dir: Path = Path("logs"),
    level: int = logging.INFO,
) -> Path:
    """Configure the ``etl`` namespace logger for this process invocation.

    Creates ``logs_dir`` if it does not exist and opens a new timestamped
    log file so each run is isolated. The console handler always prints to
    stdout with colour.

    Args:
        logs_dir: Directory where per-run log files are written.
        level: Minimum log level for both handlers (default INFO).

    Returns:
        Path to the log file created for this run.
    """
    etl_logger = logging.getLogger("etl")

    if etl_logger.handlers:
        return Path(etl_logger.handlers[-1].baseFilename  # type: ignore[attr-defined]
                    if hasattr(etl_logger.handlers[-1], "baseFilename")
                    else "logs/etl.log")

    etl_logger.setLevel(level)
    etl_logger.propagate = False

    plain_fmt = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    color_fmt = _ColorFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(color_fmt)

    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"etl_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(plain_fmt)

    etl_logger.addHandler(console)
    etl_logger.addHandler(file_handler)

    return log_file
