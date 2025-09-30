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
) -> int:
    """
    Atomically write entries using rocks-shim batch APIs.
    
    Args:
        db: RocksDB database handle
        pending_data: Mapping of keys (str or bytes) to byte values
        
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
        with db.write_batch() as wb:
            for k, v in pending_data.items():
                wb.put(_coerce_key(k), v)
        # Context manager automatically commits the batch

        logger.info("Batch complete: %s entries", f"{n:,}")
        return n

    except Exception:
        logger.exception("Error writing batch of %s entries", f"{n:,}")
        raise