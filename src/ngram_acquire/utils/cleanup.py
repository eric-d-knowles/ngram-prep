"""Safe cleanup utilities for RocksDB directories."""
from __future__ import annotations

import inspect
import logging
import shutil
import stat
import time
import uuid
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

__all__ = ["safe_db_cleanup"]


def _chmod_w(path: Path) -> None:
    """Make path writable (best-effort)."""
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    except Exception:
        pass


def _on_rm_error(func, path_str, exc_info):
    """
    Error handler for shutil.rmtree: make writable then retry once.

    If that fails, try unlinking files directly as a last resort.
    """
    p = Path(path_str)
    _chmod_w(p)
    try:
        func(path_str)
    except Exception:
        # Last resort: unlink files directly, ignore directories
        try:
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
        except Exception:
            pass


def _purge_nfs_placeholders(root: Path) -> None:
    """
    Remove NFS placeholder files (.nfsXXXXXXXX) created by open file handles.

    On NFS, removing a file with open handles creates temporary .nfs* files.
    If unlink fails (file still in use), move them outside the tree for later cleanup.
    """
    moved: list[Path] = []
    for nfs in root.glob("**/.nfs*"):
        try:
            nfs.unlink()
            logger.debug("Unlinked NFS temp: %s", nfs)
        except OSError:
            # File still in use - move it out of the way
            dest = root.parent / f".nfs_cleanup_{uuid.uuid4().hex}_{nfs.name}"
            try:
                nfs.replace(dest)
                moved.append(dest)
                logger.debug("Moved busy NFS temp: %s moved to %s", nfs, dest)
            except OSError as move_exc:
                logger.debug("Failed to move NFS temp %s: %s", nfs, move_exc)

    # Best-effort cleanup of moved placeholders
    for p in moved:
        try:
            p.unlink()
        except OSError:
            pass


def _rmtree_once(path: Path) -> None:
    """
    Call shutil.rmtree with onerror handler if supported.

    Uses introspection to detect if onerror parameter is available
    (some test mocks may not support it).
    """
    try:
        params = inspect.signature(shutil.rmtree).parameters
        if "onerror" in params:
            shutil.rmtree(path, onerror=_on_rm_error)
            return
    except (ValueError, TypeError):
        # Builtins/C-callables may lack introspection
        pass

    # Fallback: no onerror support
    shutil.rmtree(path)


def _force_rmtree(path: Path) -> None:
    """Robust rmtree with NFS placeholder purge and permission fixes."""
    if not path.exists():
        return
    _purge_nfs_placeholders(path)
    _rmtree_once(path)


def safe_db_cleanup(
        db_path: Union[str, Path],
        max_retries: int = 5,
        delay_seconds: float = 2.0,
        backoff: float = 1.5,
) -> bool:
    """
    Safely remove a RocksDB directory with retry logic.

    Handles permissions issues and NFS placeholder files (.nfs*) that
    can prevent directory removal when files have open handles.

    Strategy:
        1. If path doesn't exist, return True
        2. If path is a symlink, unlink it
        3. Use robust rmtree with retries:
           - Fix permissions via onerror handler
           - Purge NFS placeholders before each attempt
           - Exponential backoff between retries

    Args:
        db_path: Path to RocksDB directory to remove
        max_retries: Maximum number of removal attempts
        delay_seconds: Initial delay between retries
        backoff: Multiplier for exponential backoff

    Returns:
        True if path was successfully removed, False otherwise

    Raises:
        ValueError: If path exists but is not a directory

    Examples:
        >>> safe_db_cleanup("/data/my_db.rocksdb")
        True
        >>> safe_db_cleanup("/data/busy_db.rocksdb", max_retries=10)
        False  # Could not remove after 10 attempts
    """
    path = Path(db_path).expanduser().resolve()

    # Already gone
    if not path.exists():
        return True

    # Handle symlinks
    if path.is_symlink():
        try:
            path.unlink()
            return True
        except OSError as e:
            logger.error("Failed to unlink symlink %s: %s", path, e)
            return False

    # Validate it's a directory
    if not path.is_dir():
        raise ValueError(f"{path} exists but is not a directory")

    # Robust removal with retries
    delay = delay_seconds
    for attempt in range(1, max_retries + 1):
        try:
            _force_rmtree(path)
            logger.info("Successfully removed database directory (attempt %d)", attempt)
            return True
        except OSError as exc:
            if attempt == max_retries:
                logger.error(
                    "Failed to remove database directory after %d attempts: %s",
                    max_retries, exc
                )
                return False

            logger.warning(
                "Database cleanup attempt %d/%d failed: %s",
                attempt, max_retries, exc
            )
            time.sleep(delay)
            delay *= backoff

    return False