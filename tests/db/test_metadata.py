# tests/db/test_metadata.py
from __future__ import annotations

import logging

import pytest

from ngram_prep.db.metadata import (
    PROCESSED_PREFIX,
    processed_key,
    is_file_processed,
    mark_file_as_processed,
    get_processed_files,
)


class FakeDB:
    """Minimal mapping-like stand-in for rocksdict.Rdict."""
    def __init__(self):
        self.store: dict[bytes, bytes] = {}

    def __contains__(self, key: bytes) -> bool:  # type: ignore[override]
        return key in self.store

    def __setitem__(self, key: bytes, value: bytes) -> None:  # type: ignore[override]
        self.store[key] = value

    def keys(self):  # rocksdict exposes .keys(); list is fine for tests
        return list(self.store.keys())


def test_processed_key_uses_prefix():
    name = "eng-us-2-00001-of-00024.gz"
    key = processed_key(name)
    assert key.startswith(PROCESSED_PREFIX)
    assert key == PROCESSED_PREFIX + name.encode("utf-8")


def test_mark_and_check_processed(caplog):
    caplog.set_level(logging.INFO)
    db = FakeDB()

    assert not is_file_processed(db, "a.gz")
    mark_file_as_processed(db, "a.gz")

    assert processed_key("a.gz") in db  # key exists
    assert is_file_processed(db, "a.gz") is True
    assert any("Marked processed: a.gz" in r.getMessage() for r in caplog.records)


def test_get_processed_files_scans_only_prefixed_keys():
    db = FakeDB()
    db[PROCESSED_PREFIX + b"f1.gz"] = b"1"
    db[b"some/other/key"] = b"x"
    db[PROCESSED_PREFIX + b"f2.gz"] = b"1"

    got = get_processed_files(db)
    assert got == {"f1.gz", "f2.gz"}


def test_is_file_processed_handles_db_errors(caplog):
    caplog.set_level(logging.WARNING)

    class BadContains(FakeDB):
        def __contains__(self, key):  # type: ignore[override]
            raise RuntimeError("boom")

    db = BadContains()
    assert is_file_processed(db, "x.gz") is False
    assert any("Processed check failed" in r.getMessage() for r in caplog.records)


def test_get_processed_files_handles_db_errors(caplog):
    caplog.set_level(logging.WARNING)

    class BadKeys(FakeDB):
        def keys(self):  # type: ignore[override]
            raise RuntimeError("nope")

    db = BadKeys()
    assert get_processed_files(db) == set()
    assert any("Could not enumerate processed files" in r.getMessage()
               for r in caplog.records)
