# common_db/api.py
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Iterator, Tuple, Union
import rocks_shim as rs

PathLike = Union[str, Path]
KV = Tuple[bytes, bytes]

@contextmanager
def open_db(
    path: Path | str,
    *,
    mode: str = "rw",
    profile: Optional[str] = None,
    create_if_missing: Optional[bool] = None,
    error_if_exists: Optional[bool] = None,   # reserved
    merge_operator: Optional[str] = None,     # reserved
):
    kwargs = {}
    if profile is not None:
        kwargs["profile"] = profile
    if create_if_missing is not None:
        kwargs["create_if_missing"] = create_if_missing
    db = rs.open(str(path), mode=mode, **kwargs)
    try:
        yield db
    finally:
        db.close()

def prefix_scan(db: rs.DB, prefix: bytes) -> Iterator[KV]:
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
        # Iterator has no .close(); drop it so C++ dtor runs before DB closes
        del it

def range_scan(
    db: rs.DB,
    lower: bytes = b"",
    upper_exclusive: Optional[bytes] = None,
) -> Iterator[KV]:
    # normalize inputs (optional)
    if not isinstance(lower, (bytes, bytearray, memoryview)):
        raise TypeError("lower must be bytes-like")
    if isinstance(lower, (bytearray, memoryview)):
        lower = bytes(lower)
    if upper_exclusive is not None and isinstance(upper_exclusive, (bytearray, memoryview)):
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
        # important: release iterator before DB context exits
        del it

def scan_all(db: rs.DB) -> Iterator[KV]:
    """Full forward scan."""
    return range_scan(db, b"", None)
