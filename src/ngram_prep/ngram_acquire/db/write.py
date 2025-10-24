"""Batch writing utilities for RocksDB."""
from __future__ import annotations

import logging
from typing import Mapping, Union

import rocks_shim as rs

logger = logging.getLogger(__name__)

DEFAULT_WRITE_BATCH_SIZE = 50_000
__all__ = ["DEFAULT_WRITE_BATCH_SIZE", "write_batch_to_db"]


def _coerce_key(k: Union[str, bytes]) -> bytes:
    """Convert string keys to bytes using UTF-8 encoding."""
    return k if isinstance(k, bytes) else k.encode("utf-8")


def write_batch_to_db(
        db: rs.DB,
        pending_data: Mapping[Union[str, bytes], bytes],
        disable_wal: bool = True,
        metadata_keys: list[bytes] = None,
) -> int:
    """
    Atomically write entries using rocks-shim batch APIs.

    Args:
        db: RocksDB database handle
        pending_data: Mapping of keys (str or bytes) to byte values
        disable_wal: If True, disable write-ahead log for better performance
        metadata_keys: Optional list of metadata keys to mark (e.g., processed files)

    Returns:
        Number of entries written

    Raises:
        RuntimeError: If batch writing fails
    """
    if not pending_data:
        return 0

    n = len(pending_data)
    logger.info("Writing batch: %s entries", f"{n:,}")

    try:
        with db.write_batch(disable_wal=disable_wal, sync=False) as wb:
            for k, v in pending_data.items():
                wb.put(_coerce_key(k), v)

            # Add metadata markers in the same batch
            if metadata_keys:
                for meta_key in metadata_keys:
                    wb.put(meta_key, b"1")
        # Context manager automatically commits the batch

        logger.info("Batch complete: %s entries", f"{n:,}")
        return n

    except Exception:
        logger.exception("Error writing batch of %s entries", f"{n:,}")
        raise