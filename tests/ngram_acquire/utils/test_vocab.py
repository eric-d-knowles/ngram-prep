# tests/vocab/test_vocab.py
from __future__ import annotations

import struct
from typing import Dict, Tuple, List

import pytest

# Module under test
import ngram_filter.pipeline.whitelist as vocab


# -----------------------------
# Shim-style DB & iterator fakes
# -----------------------------

class _Iter:
    """rocks-shim-like iterator: seek(lower), valid(), key(), value(), next()."""
    def __init__(self, items: List[Tuple[bytes, bytes]]):
        self._items = sorted(items, key=lambda kv: kv[0])
        self._i = 0

    def seek(self, lower: bytes) -> None:
        i = 0
        while i < len(self._items) and self._items[i][0] < lower:
            i += 1
        self._i = i

    def valid(self) -> bool:
        return self._i < len(self._items)

    def key(self) -> bytes:
        return self._items[self._i][0]

    def value(self) -> bytes:
        return self._items[self._i][1]

    def next(self) -> None:
        self._i += 1


class FakeShimDB:
    """Minimal read-only store with a rocks-shim iterator interface."""
    def __init__(self, initial: Dict[bytes, bytes]):
        self._store = dict(initial)

    def iterator(self) -> _Iter:
        return _Iter(list(self._store.items()))


# -------------
# Test helpers
# -------------

FMT = vocab.FMT
TUPLE_SIZE = struct.calcsize(FMT)


def pack(*triples: Tuple[int, int, int]) -> bytes:
    """Pack a sequence of (year, match, vol) into bytes using FMT."""
    return b"".join(struct.pack(FMT, y, m, v) for (y, m, v) in triples)


def _build_small_db() -> FakeShimDB:
    """
    Create a small DB with:
      - b"apple": matches sum to 30
      - b"banana": matches sum to 18
      - b"__meta": metadata (should be skipped)
      - b"cédez": utf-8 key to test decoding behavior
    """
    return FakeShimDB(
        {
            b"apple": pack((2000, 10, 1), (2001, 20, 2)),   # total=30
            b"banana": pack((1999, 5, 1), (2000, 13, 1)),   # total=18
            b"__meta": pack((1900, 999, 1)),                # skipped
            "cédez".encode("utf-8"): pack((2010, 7, 1)),    # total=7
        }
    )


# --------------------------------
# NumPy and struct path consistency
# --------------------------------

def test_total_matches_struct_path(monkeypatch):
    # Force struct path by disabling numpy in the module
    monkeypatch.setattr(vocab, "np", None, raising=False)

    db = _build_small_db()
    # Use private helpers to compute totals
    totals = {k: vocab._total_matches(v) for (k, v) in db.iterator()._items}
    # "__meta" is present in store but _iter_db_items should skip it;
    # here we just verify _total_matches correctness on raw values:
    assert totals[b"apple"] == 30
    assert totals[b"banana"] == 18


@pytest.mark.skipif(vocab.np is None, reason="NumPy not available")
def test_total_matches_numpy_path():
    # If NumPy is available, ensure it agrees with struct calculation
    db = _build_small_db()
    vals = dict(db.iterator()._items)
    assert vocab._total_matches_numpy(vals[b"apple"]) == 30
    assert vocab._total_matches_numpy(vals[b"banana"]) == 18
    # and the generic wrapper uses NumPy without error
    assert vocab._total_matches(vals[b"apple"]) == 30


# -----------------------
# Ranking and Top-K logic
# -----------------------

def test_rank_all_decodes_and_sorts(monkeypatch):
    db = _build_small_db()
    rows = vocab._rank_all(db, decode=True)
    # Sorted descending by total
    assert rows == [("apple", 30), ("banana", 18), ("cédez", 7)]


def test_rank_all_bytes_when_decode_false(monkeypatch):
    db = _build_small_db()
    rows = vocab._rank_all(db, decode=False)
    # Keys are bytes
    assert rows[0][0] == b"apple"
    assert rows[-1][0] == "cédez".encode("utf-8")


def test_top_k_streaming(monkeypatch):
    db = _build_small_db()
    rows = vocab._top_k(db, 2, decode=True)
    assert rows == [("apple", 30), ("banana", 18)]
    # k<=0 yields empty list
    assert vocab._top_k(db, 0, decode=True) == []


# ----------------------------
# End-to-end write_vocab tests
# ----------------------------

def test_write_vocab_all(tmp_path, monkeypatch):
    db = _build_small_db()
    out = tmp_path / "vocab_all.txt"
    p = vocab.write_vocab(db, out, decode=True, sep="\t")  # path or DB both OK
    assert p == out.resolve()
    content = out.read_text(encoding="utf-8").strip().splitlines()
    assert content == ["apple\t30", "banana\t18", "cédez\t7"]


def test_write_vocab_top_k_and_no_decode(tmp_path):
    db = _build_small_db()
    out = tmp_path / "vocab_top.txt"
    p = vocab.write_vocab(db, out, top=1, decode=False, sep="|")
    assert p == out.resolve()
    # When decode=False, write_vocab still writes strings by decoding with backslashreplace
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    token, total = lines[0].split("|")
    assert token == "apple"  # ascii-safe; backslashreplace matters for non-ascii keys
    assert total == "30"


def test_write_vocab_open_by_path_uses_open_db(monkeypatch, tmp_path):
    """
    Ensure that when a filesystem path is passed, the module calls open_db(...)
    with (mode='ro', profile='read') and consumes an iterator as expected.
    """
    # Build a fake DB and fake open_db context manager
    fake_db = _build_small_db()

    calls: list[tuple[str, str, str]] = []

    class _Ctx:
        def __init__(self, path, *, mode="rw", profile=None, **_k):
            calls.append((str(path), mode, profile or ""))
        def __enter__(self):
            return fake_db
        def __exit__(self, exc_type, exc, tb):
            return False

    # Patch the module's open_db symbol
    monkeypatch.setattr(vocab, "open_db", lambda path, **kw: _Ctx(path, **kw), raising=True)

    db_dir = tmp_path / "dbdir"
    db_dir.mkdir()
    out = tmp_path / "out.txt"

    # Pass the directory path (or file path) as the DB location
    p = vocab.write_vocab(db_dir, out, decode=True)
    assert p == out.resolve()

    # Verify open_db was called with read-only + read profile
    assert calls and calls[0][1] == "ro"
    assert calls[0][2] == "read"

    # Output content is correct
    assert out.read_text(encoding="utf-8").splitlines()[0].startswith("apple\t30")
