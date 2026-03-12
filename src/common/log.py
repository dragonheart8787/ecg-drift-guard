"""Centralised logging configuration with console + optional file output."""

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Return a logger.  Configures root handler once (console + optional file)."""
    global _CONFIGURED
    if not _CONFIGURED:
        fmt = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logging.root.addHandler(console)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_file), encoding="utf-8")
            fh.setFormatter(fmt)
            logging.root.addHandler(fh)

        logging.root.setLevel(level)
        _CONFIGURED = True
    return logging.getLogger(name)


def setup_file_logging(log_file: str | Path) -> None:
    """Add a file handler to root logger (call after first get_logger)."""
    fmt = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(fmt)
    logging.root.addHandler(fh)
