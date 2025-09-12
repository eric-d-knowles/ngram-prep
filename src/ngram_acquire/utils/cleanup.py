# ngram_acquire/utils/cleanup.py
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


def _chmod_w(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    except Exception:
        # Best-effort
        pass


def _on_rm_error(func, path_str, exc_info):
    """shutil.rmtree onerror handler: make writable then retry once."""
    p = Path(path_str)
    _chmod_w(p)
    try:
        func(path_str)
    except Exception:
        # Last resort: unlink files, ignore dirs (rmtree will loop back)
        try:
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
        except Exception:
            pass


def _purge_nfs_placeholders(root: Path) -> None:
    """
    On NFS, removing a file with open handles can leave .nfsXXXXXXXX temp files.
    Try to unlink; if that fails, move them out of the tree and unlink later.
    """
    moved: list[Path] = []
    for nfs in root.glob("**/.nfs*"):
        try:
            nfs.unlink()
            logger.debug("Unlinked NFS temp: %s", nfs)
        except OSError:
            dest = root.parent / f".nfs_cleanup_{uuid.uuid4().hex}_{nfs.name}"
            try:
                nfs.replace(dest)
                moved.append(dest)
                # Tests expect the phrase "moved to"
                logger.debug("Moved busy NFS temp: %s moved to %s", nfs, dest)
            except OSError as move_exc:
                logger.debug("Failed to move NFS temp %s: %s", nfs, move_exc)

    # Best-effort unlink of moved placeholders
    for p in moved:
        try:
            p.unlink()
        except OSError:
            pass


def _rmtree_once(path: Path) -> None:
    """
    Call shutil.rmtree exactly once, detecting whether the current object
    supports 'onerror' (tests monkeypatch rmtree without it).
    """
    try:
        params = inspect.signature(shutil.rmtree).parameters
        if "onerror" in params:
            shutil.rmtree(path, onerror=_on_rm_error)
            return
    except (ValueError, TypeError):
        # Builtins / C-callables / lambdas may lack introspection
        pass
    # Fallback: no onerror supported by current rmtree
    shutil.rmtree(path)


def _force_rmtree(path: Path) -> None:
    """Robust rmtree with chmod+retry + NFS placeholder purge."""
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
    Remove a RocksDB directory safely, handling permissions and NFS .nfs* files.

    Strategy
    --------
    1) If missing: return True.
    2) If it's a symlink: unlink and return True.
    3) In-place robust rmtree with retries:
       - chmod via onerror handler (when supported)
       - purge .nfs* placeholders before each attempt

    Returns
    -------
    True  : The path was removed.
    False : Could not remove the path after all retries.
    """
    path = Path(db_path).expanduser()

    # 1) Not present → done
    if not path.exists():
        return True

    # 2) If someone accidentally made a file, fail loudly
    if path.is_symlink():
        try:
            path.unlink()
            return True
        except OSError as e:
            logger.error("Failed to unlink symlink %s: %s", path, e)
            return False
    if not path.is_dir():
        raise ValueError(f"{path!s} exists but is not a directory")

    # 3) In-place robust rmtree with retries
    delay = delay_seconds
    for attempt in range(1, max_retries + 1):
        try:
            _force_rmtree(path)
            logger.info("Successfully removed database dir (attempt %d)", attempt)
            return True
        except OSError as exc:
            if attempt == max_retries:
                logger.error(
                    "Failed to remove database dir after %d attempts: %s",
                    max_retries, exc,
                )
                return False
            logger.warning(
                "Database cleanup attempt %d/%d failed: %s",
                attempt, max_retries, exc,
            )
            time.sleep(delay)
            delay *= backoff

    return False
