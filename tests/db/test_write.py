# tests/db/test_write.py
from __future__ import annotations

import logging

import pytest

from ngram_prep.db.write import write_batch_to_db


class FakeWriteBatch:
    def __init__(self):
        self.ops: list[tuple[bytes, bytes]] = []

    def put(self, key: bytes, val: bytes) -> None:
        assert isinstance(key, (bytes, bytearray))
        assert isinstance(val, (bytes, bytearray))
        self.ops.append((bytes(key), bytes(val)))


class FakeDB:
    def __init__(self):
        self.batches: list[list[tuple[bytes, bytes]]] = []
        self.fail = False

    def write(self, wb) -> None:
        if self.fail:
            raise RuntimeError("db down")
        self.batches.append(list(getattr(wb, "ops", [])))


def test_returns_zero_for_empty_batch():
    db = FakeDB()
    assert write_batch_to_db(db, {}) == 0
    assert db.batches == []


def test_writes_entries_and_encodes_keys(monkeypatch, caplog):
    import ngram_prep.db.write as write_mod
    monkeypatch.setattr(write_mod, "WriteBatch", FakeWriteBatch, raising=True)

    db = FakeDB()
    caplog.set_level(logging.INFO)

    data = {"alpha": b"\x01\x02", "beta": b"\x03\x04"}
    count = write_batch_to_db(db, data)
    assert count == 2

    assert db.batches and len(db.batches[0]) == 2
    keys = [k for k, _ in db.batches[0]]
    vals = [v for _, v in db.batches[0]]
    assert keys == [b"alpha", b"beta"]
    assert vals == [b"\x01\x02", b"\x03\x04"]

    msgs = [r.getMessage() for r in caplog.records]
    assert any("Writing batch: 2 entries" in m for m in msgs)
    assert any("Batch complete" in m for m in msgs)


def test_logs_and_raises_on_db_error(monkeypatch, caplog):
    import ngram_prep.db.write as write_mod
    monkeypatch.setattr(write_mod, "WriteBatch", FakeWriteBatch, raising=True)

    db = FakeDB()
    db.fail = True
    caplog.set_level(logging.ERROR)

    with pytest.raises(RuntimeError):
        write_batch_to_db(db, {"x": b"y"})

    assert any("Error writing batch" in r.getMessage() for r in caplog.records)
