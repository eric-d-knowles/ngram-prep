# ngram_acquire/db/metadata.py
from __future__ import annotations

import logging
from typing import Set

import rocks_shim as rs  # <-- rocks-shim (not python-rocksdb)

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


def is_file_processed(db: "rs.DB", filename: str) -> bool:
    """
    O(1) membership test via a per-file marker key.
    rocks-shim DB exposes .get() and returns None when not found.
    """
    try:
        return db.get(processed_key(filename)) is not None
    except Exception as exc:
        logger.warning("Processed check failed for %s: %s", filename, exc)
        return False


def mark_file_as_processed(db: "rs.DB", filename: str) -> None:
    """
    Atomic per-file mark; safe with concurrent writers (RocksDB key-level atomicity).
    """
    try:
        db.put(processed_key(filename), b"1")
        logger.info("Marked processed: %s", filename)
    except Exception as exc:
        logger.error("Could not mark file as processed %s: %s", filename, exc)


def get_processed_files(db: "rs.DB") -> set[str]:
    """
    Enumerate processed files by scanning keys with the processed prefix.

    Uses a forward iterator starting at PROCESSED_PREFIX and stops when keys
    no longer share the prefix. Prefer is_file_processed() in hot paths.
    """
    out: Set[str] = set()
    try:
        it = db.iterator()  # rocks-shim iterator: seek(), valid(), key(), value(), next()
        prefix = PROCESSED_PREFIX
        it.seek(prefix)
        while it.valid():
            k = it.key()
            if not k.startswith(prefix):
                break
            out.add(k[len(prefix):].decode("utf-8", "ignore"))
            it.next()
        # If your shim exposes it.close(), it’s fine to call; otherwise ignore.
        if hasattr(it, "close"):
            try:
                it.close()
            except Exception:
                pass
    except Exception as exc:
        logger.warning("Could not enumerate processed files: %s", exc)
    return out
