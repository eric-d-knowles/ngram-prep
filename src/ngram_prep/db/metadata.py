# ngram_prep/db/metadata.py
from __future__ import annotations

import logging
from rocksdict import Rdict

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
    return PROCESSED_PREFIX + filename.encode("utf-8")

def is_file_processed(db: Rdict, filename: str) -> bool:
    """O(1) membership test via a per-file marker key."""
    try:
        return processed_key(filename) in db
    except Exception as exc:
        logger.warning("Processed check failed for %s: %s", filename, exc)
        return False

def mark_file_as_processed(db: Rdict, filename: str) -> None:
    """Atomic per-file mark; safe with multiple workers."""
    try:
        db[processed_key(filename)] = b"1"
        logger.info("Marked processed: %s", filename)
    except Exception as exc:
        logger.error("Could not mark file as processed %s: %s", filename, exc)

def get_processed_files(db: Rdict) -> set[str]:
    """
    Enumerate processed files by scanning keys with the processed prefix.
    Use sparingly on huge DBs; prefer is_file_processed() in hot paths.
    """
    out: set[str] = set()
    try:
        for k in db.keys():  # Rdict exposes .keys()/.items()
            if isinstance(k, (bytes, bytearray)) and k.startswith(PROCESSED_PREFIX):
                out.add(k[len(PROCESSED_PREFIX):].decode("utf-8", "ignore"))
    except Exception as exc:
        logger.warning("Could not enumerate processed files: %s", exc)
    return out
