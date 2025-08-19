from __future__ import annotations

import logging
from typing import Mapping

from rocksdict import Rdict, WriteBatch  # type: ignore

logger = logging.getLogger(__name__)

# Tunable defaults
DEFAULT_WRITE_BATCH_SIZE = 50_000  # entries (not bytes)

__all__ = ["DEFAULT_WRITE_BATCH_SIZE", "write_batch_to_db"]


def write_batch_to_db(db: Rdict, pending_data: Mapping[str, bytes]) -> int:
    """
    Write accumulated ngrams to RocksDB as one atomic batch.

    Parameters
    ----------
    db : rocksdict.Rdict
        Open RocksDB handle.
    pending_data : Mapping[str, bytes]
        {ngram_text: packed_value_bytes}

    Returns
    -------
    int
        Number of entries written.

    Raises
    ------
    Exception
        Propagates DB errors after logging.
    """
    if not pending_data:
        return 0

    entries_count = len(pending_data)
    logger.info("Writing batch: %s entries", f"{entries_count:,}")

    wb = WriteBatch()
    try:
        # Rdict keys are bytes; encode ngram strings once here.
        for ngram_key, serialized in pending_data.items():
            wb.put(ngram_key.encode("utf-8"), serialized)
        db.write(wb)
        logger.info("Batch complete")
        return entries_count
    except Exception:
        logger.exception("Error writing batch")
        raise
    finally:
        # Ensure native resources are promptly freed
        try:
            del wb
        except Exception:
            pass
