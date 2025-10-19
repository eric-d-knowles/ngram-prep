# common_db/api.py
"""High-level API for RocksDB operations via rocks_shim C++ binding."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import rocks_shim as rs

PathLike = Union[str, Path]
KV = Tuple[bytes, bytes]


@contextmanager
def open_db(
        path: PathLike,
        *,
        mode: str = "rw",
        profile: Optional[str] = None,
        create_if_missing: Optional[bool] = None,
        error_if_exists: Optional[bool] = None,
        merge_operator: Optional[str] = None,
):
    """
    Open a RocksDB database with automatic cleanup.

    Args:
        path: Path to database directory
        mode: Access mode - 'r' (read-only), 'rw' (read-write)
        profile: Performance profile name (if supported by shim)
        create_if_missing: Create database if it doesn't exist
        error_if_exists: Error if database already exists (reserved)
        merge_operator: Custom merge operator name (reserved)
    """
    kwargs = {}
    if profile is not None:
        kwargs["profile"] = profile
    if create_if_missing is not None:
        kwargs["create_if_missing"] = create_if_missing
    # Reserved for future use
    if error_if_exists is not None:
        kwargs["error_if_exists"] = error_if_exists
    if merge_operator is not None:
        kwargs["merge_operator"] = merge_operator

    db = rs.open(str(path), mode=mode, **kwargs)
    try:
        yield db
    finally:
        # Flush memtables to SST files before closing (write modes only)
        if mode in ("rw", "w"):
            try:
                db.finalize_bulk()
            except AttributeError:
                pass  # Method not available, skip flush
            except Exception as e:
                # Suppress "No such file" errors - shard may have been deleted by async reader
                # This is safe because finalize_bulk() writes to disk synchronously before returning
                if "No such file or directory" not in str(e):
                    print(f"Warning: Database flush failed for {path}: {e}", flush=True)
        try:
            db.close()
        except Exception:
            pass  # Ignore close errors if file was deleted


def prefix_scan(db: rs.DB, prefix: bytes) -> Iterator[KV]:
    """
    Scan all key-value pairs matching a given prefix.

    Args:
        db: RocksDB database handle
        prefix: Binary prefix to match
    """
    if not isinstance(prefix, bytes):
        raise TypeError(f"prefix must be bytes, got {type(prefix).__name__}")

    it = db.iterator()
    try:
        it.seek(prefix)
        while it.valid():
            k = it.key()
            if not k.startswith(prefix):
                break
            yield k, it.value()
            it.next()
    finally:
        del it


def range_scan(
        db: rs.DB,
        lower: bytes = b"",
        upper_exclusive: Optional[bytes] = None,
) -> Iterator[KV]:
    """
    Scan key-value pairs in range [lower, upper_exclusive).

    Args:
        db: RocksDB database handle
        lower: Inclusive lower bound (default: scan from start)
        upper_exclusive: Exclusive upper bound (default: scan to end)
    """
    # Normalize bytes-like inputs to bytes
    if not isinstance(lower, (bytes, bytearray, memoryview)):
        raise TypeError(f"lower must be bytes-like, got {type(lower).__name__}")
    if isinstance(lower, (bytearray, memoryview)):
        lower = bytes(lower)

    if upper_exclusive is not None:
        if not isinstance(upper_exclusive, (bytes, bytearray, memoryview)):
            raise TypeError(f"upper_exclusive must be bytes-like, got {type(upper_exclusive).__name__}")
        if isinstance(upper_exclusive, (bytearray, memoryview)):
            upper_exclusive = bytes(upper_exclusive)

    it = db.iterator()
    try:
        it.seek(lower)
        while it.valid():
            k = it.key()
            if upper_exclusive is not None and k >= upper_exclusive:
                break
            yield k, it.value()
            it.next()
    finally:
        del it


def scan_all(db: rs.DB) -> Iterator[KV]:
    """Full forward scan of all key-value pairs."""
    return range_scan(db, b"", None)