# ngram_prep/db/rocks.py
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from rocksdict import Options, Rdict

logger = logging.getLogger(__name__)


def make_default_options(
    *,
    create_if_missing: bool = True,
    background_jobs: Optional[int] = None,
    write_buffer_size: int = 256 * 1024 * 1024,
    l0_compaction_trigger: int = 12,
) -> Options:
    """
    Sensible defaults tuned for throughput. Adjust as needed per workload.
    """
    opts = Options()
    if create_if_missing:
        opts.create_if_missing(True)

    if background_jobs is None:
        # Heuristic: at least 4, else scale with cores
        background_jobs = max(4, (os.cpu_count() or 4))
    opts.set_max_background_jobs(int(background_jobs))

    opts.set_write_buffer_size(write_buffer_size)
    opts.set_level_zero_file_num_compaction_trigger(l0_compaction_trigger)
    return opts


def setup_rocksdb(
    db_path: str | Path,
    *,
    options: Optional[Options] = None,
    retries: int = 1,
    delay_seconds: float = 0.2,
    backoff: float = 2.0,
) -> Rdict:
    """
    Open a RocksDB at `db_path` with tuned options, creating parents as needed.

    Retries only on lock-related errors (common after crashed processes).

    Parameters
    ----------
    db_path : str | Path
        Directory for the RocksDB instance.
    options : rocksdict.Options | None
        If None, uses `make_default_options()`.
    retries : int
        Additional attempts after the first (total tries = retries + 1).
    delay_seconds : float
        Initial sleep between retries.
    backoff : float
        Multiplicative backoff factor for subsequent sleeps.

    Returns
    -------
    rocksdict.Rdict
    """
    path = Path(db_path).expanduser()
    # Ensure parent directory exists (RocksDB will create the DB dir itself)
    path.parent.mkdir(parents=True, exist_ok=True)

    opts = options or make_default_options()

    attempt = 1
    delay = delay_seconds
    while True:
        try:
            db = Rdict(str(path), opts)
            logger.info("Opened RocksDB at %s (attempt %d)", path, attempt)
            return db
        except Exception as exc:
            msg = str(exc).lower()
            lock_issue = "lock" in msg
            if not lock_issue or attempt > retries:
                logger.error("Failed to open RocksDB at %s: %s", path, exc)
                raise

            logger.warning(
                "RocksDB lock issue opening %s (attempt %d/%d): %s",
                path,
                attempt,
                retries + 1,
                exc,
            )
            time.sleep(delay)
            delay *= backoff
            attempt += 1
