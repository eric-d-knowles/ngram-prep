# ngram_prep/io/cleanup.py
from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def safe_db_cleanup(
    db_path: Union[str, Path],
    max_retries: int = 5,
    delay_seconds: float = 2.0,
    backoff: float = 1.5,
) -> bool:
    """
    Remove a RocksDB directory safely, handling lingering NFS temp files.

    Behavior
    --------
    - If the path doesn't exist: returns True (idempotent no-op).
    - If the path exists but isn't a directory: raises ValueError.
    - Handles NFS .nfs* placeholders by unlinking or renaming them to the
      parent when busy, then removing the directory.
    """
    path = Path(db_path).expanduser()

    if not path.exists():
        return True
    if not path.is_dir():
        raise ValueError(f"{path!s} exists but is not a directory")

    def _best_effort_unlink(paths: list[Path]) -> None:
        for p in paths:
            try:
                p.unlink()
            except OSError:
                pass

    delay = delay_seconds
    for attempt in range(1, max_retries + 1):
        moved_outside: list[Path] = []
        try:
            # Clear lingering NFS temp files
            for nfs_file in path.glob(".nfs*"):
                try:
                    nfs_file.unlink()
                    logger.debug("Removed NFS temp file: %s", nfs_file.name)
                except OSError as exc:
                    dest = (
                        path.parent
                        / f".nfs_cleanup_{uuid.uuid4().hex}_{nfs_file.name}"
                    )
                    try:
                        nfs_file.rename(dest)
                        moved_outside.append(dest)
                        logger.debug(
                            "Could not unlink %s (%s); moved to %s",
                            nfs_file.name,
                            exc,
                            dest.name,
                        )
                    except OSError as move_exc:
                        logger.debug(
                            "Failed to move %s out of dir (%s)",
                            nfs_file.name,
                            move_exc,
                        )

            shutil.rmtree(path)
            logger.info("Successfully removed database (attempt %d)", attempt)

            _best_effort_unlink(moved_outside)
            return True

        except OSError as exc:
            _best_effort_unlink(moved_outside)

            if attempt == max_retries:
                logger.error(
                    "Failed to remove database after %d attempts: %s",
                    max_retries,
                    exc,
                )
                return False

            logger.warning(
                "Database cleanup attempt %d/%d failed: %s",
                attempt,
                max_retries,
                exc,
            )
            logger.debug(
                "Retrying in %.2f seconds (backoff=%.2f)...",
                delay,
                backoff,
            )
            time.sleep(delay)
            delay *= backoff

    return False
