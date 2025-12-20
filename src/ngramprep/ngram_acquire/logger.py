"""Logging configuration for ngram acquisition pipeline."""
from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union

__all__ = ["setup_logger"]


def setup_logger(
        db_path: Union[str, Path],
        *,
        level: int = logging.INFO,
        filename_prefix: str = "ngram_download",
        console: bool = False,
        rotate: bool = False,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 3,
        force: bool = False,
) -> Path:
    """
    Configure root logging to write to a file in/near the database directory.

    Creates a timestamped log file in the database directory (or its parent if
    db_path is a file). Optionally adds console output and log rotation.

    Args:
        db_path: Path to database (directory or file)
        level: Logging level (default: INFO)
        filename_prefix: Prefix for log filename
        console: If True, also log to console
        rotate: If True, use RotatingFileHandler instead of FileHandler
        max_bytes: Maximum log file size before rotation (if rotate=True)
        backup_count: Number of backup files to keep (if rotate=True)
        force: If True, remove existing handlers before adding new ones

    Returns:
        Path to the created log file

    Examples:
        >>> log_path = setup_logger("/data/ngrams/1grams.db")
        >>> log_path
        PosixPath('/data/ngrams/ngram_download_20250929_175430.log')
    """
    db_path = Path(db_path).expanduser().resolve()

    # Determine log directory
    # If db_path is a directory (or has no suffix, suggesting RocksDB dir),
    # use it directly. Otherwise use parent directory.
    if db_path.is_dir() or not db_path.suffix:
        log_dir = db_path
    else:
        log_dir = db_path.parent

    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{filename_prefix}_{timestamp}.log"

    # Configure root logger
    root = logging.getLogger()

    # Remove existing handlers if force=True
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    root.setLevel(level)

    # Consistent formatter for all handlers
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler (with optional rotation)
    if rotate:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    else:
        file_handler = logging.FileHandler(
            log_path,
            mode="w",
            encoding="utf-8",
        )

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Optional console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    root.info("Logging initialized: %s", log_path)
    return log_path