"""
Logging configuration for the NER system.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
) -> None:
    """
    Configure root logger.

    Parameters
    ----------
    level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file: Optional path to write logs to a file (in addition to stdout).
    fmt: Log record format string.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler (optional)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)