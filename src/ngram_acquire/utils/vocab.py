# ngram_acquire/vocab.py
from __future__ import annotations

import heapq
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Tuple, Union

try:
    import numpy as np  # optional fast path
except Exception:
    np = None  # type: ignore[assignment]

import rocks_shim as rs  # <-- rocks-shim only
from common_db.api import open_db  # read profile via rocks-shim

FMT = "<QQQ"                 # (year, match_count, volume_count), little-endian u64s
TUPLE_SIZE = struct.calcsize(FMT)
METADATA_PREFIX = b"__"      # skip metadata keys like "__..."
DECODING = "utf-8"           # decoding for keys when decode=True

__all__ = [
    "write_vocab",
    # (the helpers below remain module-private; exported symbol is write_vocab)
]


@contextmanager
def _db_from(db_or_path: Union[str, Path, "rs.DB"]):
    """
    Yield an open rocks-shim DB. If a path is provided, open with read-optimized
    profile when available (profile="read").
    """
    if hasattr(db_or_path, "iterator") and callable(getattr(db_or_path, "iterator")):
        # Assume it's a rocks-shim DB-like object
        yield db_or_path  # type: ignore[misc]
    else:
        path = str(Path(db_or_path))
        # Use your rocks-shim context manager. If "read" is unsupported, shim can ignore it.
        with open_db(path, mode="ro", profile="read") as db:
            yield db


def _iter_db_items(db: "rs.DB") -> Iterator[Tuple[bytes, bytes]]:
    """
    Yield (key_bytes, value_bytes) for all non-metadata entries using rocks-shim iterator.
    rocks-shim iterator interface: it.seek(lower), it.valid(), it.key(), it.value(), it.next()
    """
    it = db.iterator()
    # Seek to start: empty lower bound (b"") is the minimal byte prefix
    it.seek(b"")
    while it.valid():
        k = it.key()
        if not k.startswith(METADATA_PREFIX):
            yield k, it.value()
        it.next()


def _total_matches_struct(value_bytes: bytes) -> int:
    """Sum match_count using struct.iter_unpack (portable path)."""
    usable = (len(value_bytes) // TUPLE_SIZE) * TUPLE_SIZE
    tot = 0
    for (_year, match, _vol) in struct.iter_unpack(FMT, value_bytes[:usable]):
        tot += match
    return tot


def _total_matches_numpy(value_bytes: bytes) -> int:
    """Sum match_count using NumPy zero-copy view (fast path)."""
    assert np is not None
    a = np.frombuffer(value_bytes, dtype="<u8")  # type: ignore[attr-defined]
    n = (a.size // 3) * 3
    return int(a[:n][1::3].sum())


def _total_matches(value_bytes: bytes) -> int:
    if np is not None:
        try:
            return _total_matches_numpy(value_bytes)
        except Exception:
            # Safety net: if numpy dtype/view fails for any reason, fall back
            pass
    return _total_matches_struct(value_bytes)


def _maybe_decode(b: bytes, decode: bool) -> Union[str, bytes]:
    return b.decode(DECODING, "replace") if decode else b


def _rank_all(
    db_or_path: Union[str, Path, "rs.DB"],
    *,
    decode: bool = True,
) -> List[Tuple[Union[str, bytes], int]]:
    """Compute totals for all keys and sort descending (RAM ~ O(N))."""
    rows: List[Tuple[Union[str, bytes], int]] = []
    with _db_from(db_or_path) as db:
        for k, v in _iter_db_items(db):
            rows.append((_maybe_decode(k, decode), _total_matches(v)))
    rows.sort(key=lambda kv: kv[1], reverse=True)
    return rows


def _top_k(
    db_or_path: Union[str, Path, "rs.DB"],
    k: int,
    *,
    decode: bool = True,
) -> List[Tuple[Union[str, bytes], int]]:
    """Streaming top-K via min-heap (RAM ~ O(K))."""
    if k <= 0:
        return []
    heap: List[Tuple[int, bytes]] = []  # (total, key_bytes)
    with _db_from(db_or_path) as db:
        for kbytes, vbytes in _iter_db_items(db):
            tot = _total_matches(vbytes)
            if len(heap) < k:
                heapq.heappush(heap, (tot, kbytes))
            elif tot > heap[0][0]:
                heapq.heapreplace(heap, (tot, kbytes))
    out: List[Tuple[Union[str, bytes], int]] = [
        (_maybe_decode(kb, decode), tot) for (tot, kb) in heap
    ]
    out.sort(key=lambda kv: kv[1], reverse=True)
    return out


def write_vocab(
    db_or_path: Union[str, Path, "rs.DB"],
    dest: Union[str, Path],
    *,
    top: int | None = None,
    decode: bool = True,
    sep: str = "\t",
) -> Path:
    """
    Write a plain TXT file of tokens ranked by total frequency (desc).
    Each line: <token><sep><total_matches>
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    rows = _top_k(db_or_path, top, decode=decode) if top is not None else _rank_all(
        db_or_path, decode=decode
    )

    with tmp.open("w", encoding="utf-8", newline="") as f:
        for token, total in rows:
            if isinstance(token, bytes):
                token_str = token.decode("utf-8", "backslashreplace")
            else:
                token_str = token
            f.write(f"{token_str}{sep}{total}\n")

    tmp.replace(dest)
    return dest.resolve()
