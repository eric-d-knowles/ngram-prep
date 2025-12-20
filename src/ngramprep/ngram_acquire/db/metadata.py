"""Metadata tracking for processed ngram files in RocksDB."""
from __future__ import annotations

import logging

import rocks_shim as rs

logger = logging.getLogger(__name__)

PROCESSED_PREFIX = b"__processed__/"

__all__ = [
    "PROCESSED_PREFIX",
    "processed_key",
    "is_file_processed",
    "mark_file_as_processed",
    "get_processed_files",
]


def processed_key(filename: str) -> bytes:
    """Generate metadata key for a processed file."""
    return PROCESSED_PREFIX + filename.encode("utf-8")


def is_file_processed(db: rs.DB, filename: str) -> bool:
    """
    Check if a file has been processed (O(1) lookup).
    
    Args:
        db: RocksDB database handle
        filename: Name of file to check
        
    Returns:
        True if file has been marked as processed
    """
    try:
        return db.get(processed_key(filename)) is not None
    except Exception:
        logger.exception("Failed to check processed status for %s", filename)
        return False


def mark_file_as_processed(db: rs.DB, filename: str) -> None:
    """
    Mark a file as processed (atomic write).
    
    Args:
        db: RocksDB database handle
        filename: Name of file to mark
    """
    try:
        db.put(processed_key(filename), b"1")
        logger.info("Marked as processed: %s", filename)
    except Exception:
        logger.exception("Failed to mark file as processed: %s", filename)
        raise


def get_processed_files(db: rs.DB) -> set[str]:
    """
    Retrieve all processed filenames via prefix scan.
    
    Scans keys with PROCESSED_PREFIX. For membership tests,
    prefer is_file_processed() instead.
    
    Args:
        db: RocksDB database handle
        
    Returns:
        Set of processed filenames
    """
    processed = set()
    it = db.iterator()
    try:
        it.seek(PROCESSED_PREFIX)
        while it.valid():
            k = it.key()
            if not k.startswith(PROCESSED_PREFIX):
                break
            filename = k[len(PROCESSED_PREFIX):].decode("utf-8", "ignore")
            processed.add(filename)
            it.next()
    except Exception:
        logger.exception("Failed to enumerate processed files")
    finally:
        del it

    return processed