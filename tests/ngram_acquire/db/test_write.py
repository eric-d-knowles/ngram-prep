# tests/db/test_writer.py
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pytest

from ngram_acquire.db.write import write_batch_to_db


# ------------------------------
# Fakes for rocks-shim behavior
# ------------------------------

class _WB:
    def __init__(self):
        self.ops: List[Tuple[bytes, bytes]] = []

    def put(self, k: bytes, v: bytes):
        self.ops.append((k, v))


class FakeShimDB_WriteBatch:
    """Exposes rs.WriteBatch + db.write(write_batch)."""
    def __init__(self):
        self.store: Dict[bytes, bytes] = {}
        self._written_batches: List[_WB] = []

    # rocks-shim expects: db.write(wb)
    def write(self, wb: _WB):
        self._written_batches.append(wb)
        for k, v in wb.ops:
            self.store[k] = v


class _CtxBatch:
    def __init__(self, outer):
        self.outer = outer
        self.ops: List[Tuple[bytes, bytes]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            for k, v in self.ops:
                self.outer.store[k] = v
        # swallow nothing; propagate exceptions if any
        return False

    def put(self, k: bytes, v: bytes):
        self.ops.append((k, v))


class FakeShimDB_BatchCtx:
    """Exposes db.batch() context manager."""
    def __init__(self):
        self.store: Dict[bytes, bytes] = {}

    def batch(self):
        return _CtxBatch(self)


class FakeShimDB_NoBatch:
    """No batch support — should cause write_batch_to_db to raise."""
    def __init__(self):
        self.store: Dict[bytes, bytes] = {}


# -----------------------------------
# Monkeypatch rocks_shim.WriteBatch
# -----------------------------------

@pytest.fixture
def patch_rocks_shim_WriteBatch(monkeypatch):
    # Provide a dummy rocks_shim.WriteBatch visible to writer.py
    import rocks_shim as rs
    monkeypatch.setattr(rs, "WriteBatch", _WB, raising=True)
    return rs


# -----------------
# Actual test cases
# -----------------

def test_writebatch_path(patch_rocks_shim_WriteBatch):
    db = FakeShimDB_WriteBatch()
    data = {"a": b"1", b"b": b"2", "c": b"3"}

    wrote = write_batch_to_db(db, data)
    assert wrote == 3

    # Keys coerced to bytes and stored
    assert db.store[b"a"] == b"1"
    assert db.store[b"b"] == b"2"
    assert db.store[b"c"] == b"3"

    # Ensure a single atomic batch invocation occurred
    assert len(db._written_batches) == 1
    assert sorted(db._written_batches[0].ops) == sorted([(b"a", b"1"), (b"b", b"2"), (b"c", b"3")])


def test_context_batch_path(monkeypatch):
    # Remove WriteBatch so the function must use db.batch()
    import rocks_shim as rs
    if hasattr(rs, "WriteBatch"):
        monkeypatch.delattr(rs, "WriteBatch", raising=False)

    db = FakeShimDB_BatchCtx()
    data = {b"k1": b"v1", "k2": b"v2"}

    wrote = write_batch_to_db(db, data)
    assert wrote == 2

    assert db.store[b"k1"] == b"v1"
    assert db.store[b"k2"] == b"v2"


def test_raises_when_no_batch_supported(monkeypatch):
    # Ensure no WriteBatch present
    import rocks_shim as rs
    if hasattr(rs, "WriteBatch"):
        monkeypatch.delattr(rs, "WriteBatch", raising=False)

    db = FakeShimDB_NoBatch()
    with pytest.raises(RuntimeError):
        write_batch_to_db(db, {b"x": b"y"})


def test_empty_map_noop(monkeypatch):
    # Works for both paths; use the no-batch fake just to ensure early return
    db = FakeShimDB_NoBatch()
    # Even though DB lacks batch API, empty input should not attempt any write
    # and thus should not raise.
    assert write_batch_to_db(db, {}) == 0
