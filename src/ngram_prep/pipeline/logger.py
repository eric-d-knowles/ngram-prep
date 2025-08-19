# ngram_prep/pipeline/logger.py
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

def setup_logger(
    db_path: str | Path,
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

    Returns the path to the log file. Safe to call once at process start.
    """
    p = Path(db_path).expanduser()

    # RocksDB path is a directory; if unsure, prefer parent when a file-like
    # path is given (has a suffix).
    log_dir = p if (p.is_dir() or not p.suffix) else p.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{filename_prefix}_{ts}.log"

    root = logging.getLogger()
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)

    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rotate:
        fhandler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
    else:
        fhandler = logging.FileHandler(log_path, mode="w", encoding="utf-8")

    fhandler.setLevel(level)
    fhandler.setFormatter(fmt)
    root.addHandler(fhandler)

    if console:
        shandler = logging.StreamHandler()
        shandler.setLevel(level)
        shandler.setFormatter(fmt)
        root.addHandler(shandler)

    root.info("Logging to: %s", str(log_path))
    return log_path
