from __future__ import annotations

import logging
from typing import Mapping, Union, Optional

import rocks_shim as rs

logger = logging.getLogger(__name__)

DEFAULT_WRITE_BATCH_SIZE = 50_000
__all__ = ["DEFAULT_WRITE_BATCH_SIZE", "write_batch_to_db"]


def _coerce_key(k: Union[str, bytes]) -> bytes:
    return k if isinstance(k, bytes) else str(k).encode("utf-8")


def _db_write_with_batch(db: "rs.DB", wb: object) -> None:
    """
    Strictly apply a WriteBatch using common method names.
    Raises if none are present.
    """
    for name in ("write_batch", "write", "writebatch", "apply"):
        fn = getattr(db, name, None)
        if callable(fn):
            fn(wb)  # type: ignore[misc]
            return
    raise RuntimeError(
        "rocks-shim DB lacks a recognized batch entrypoint "
        "(expected one of: write_batch, write, writebatch, apply)"
    )


def _resolve_default_cf(db: object):
    """
    Try to locate a default column family handle on the DB for put_cf().
    Common shims expose one of these attributes.
    """
    for attr in ("default_cf", "default_cf_handle", "default_column_family", "default_column_family_handle"):
        cf = getattr(db, attr, None)
        if cf is not None:
            return cf
    # Some shims keep CFs in a dict/attr; add more discovery here if needed.
    raise RuntimeError("WriteBatch.put_cf is available but no default column-family handle was found on DB")


def write_batch_to_db(
        db: "rs.DB",
        pending_data: Mapping[Union[str, bytes], bytes],
) -> int:
    """
    Atomically write entries using rocks-shim batch APIs.
    Uses DB.write_batch() method to create WriteBatch objects.
    """
    if not pending_data:
        return 0

    n = len(pending_data)
    logger.info("Writing batch: %s entries", f"{n:,}")

    try:
        # Create WriteBatch through the DB object (not rs.WriteBatch())
        with db.write_batch() as wb:
            for k, v in pending_data.items():
                wb.put(_coerce_key(k), v)
        # Context manager automatically commits the batch

        logger.info("Batch complete: %s entries", f"{n:,}")
        return n

    except Exception:
        logger.exception("Error writing batch")
        raise
