# tests/db/test_metadata.py
from __future__ import annotations

import logging
from typing import Optional

import pytest

from ngram_acquire.db.metadata import (
    PROCESSED_PREFIX,
    processed_key,
    is_file_processed,
    mark_file_as_processed,
    get_processed_files,
)

# -----------------------------------------------------------------------------
# rocks-shim-compatible fakes
# -----------------------------------------------------------------------------

class _FakeIter:
    """rocks-shim-style iterator: seek(), valid(), key(), value(), next()."""
    def __init__(self, store: dict[bytes, bytes]):
        self._pairs = sorted(store.items())  # list[(k, v)], sorted by k
        self._i = 0

    def seek(self, lower: bytes) -> None:
        i = 0
        n = len(self._pairs)
        # linear scan is fine for tests (could bisect)
        while i < n and self._pairs[i][0] < lower:
            i += 1
        self._i = i

    def valid(self) -> bool:
        return 0 <= self._i < len(self._pairs)

    def key(self) -> bytes:
        return self._pairs[self._i][0]

    def value(self) -> bytes:
        return self._pairs[self._i][1]

    def next(self) -> None:
        self._i += 1


class FakeShimDB:
    """
    Minimal stand-in for rocks_shim.DB exposing:
      - get(key) -> bytes|None
      - put(key, value) -> None
      - iterator() -> _FakeIter
    """
    def __init__(self):
        self._store: dict[bytes, bytes] = {}

    def get(self, key: bytes) -> Optional[bytes]:
        return self._store.get(key)

    def put(self, key: bytes, value: bytes) -> None:
        self._store[key] = value

    def iterator(self):
        return _FakeIter(self._store)


# -----------------------------------------------------------------------------
# Unit tests against the fakes (fast, deterministic)
# -----------------------------------------------------------------------------

def test_processed_key_uses_prefix():
    name = "eng-us-2-00001-of-00024.gz"
    key = processed_key(name)
    assert key.startswith(PROCESSED_PREFIX)
    assert key == PROCESSED_PREFIX + name.encode("utf-8")


def test_mark_and_check_processed(caplog):
    caplog.set_level(logging.INFO)
    db = FakeShimDB()

    assert not is_file_processed(db, "a.gz")
    mark_file_as_processed(db, "a.gz")

    assert db.get(processed_key("a.gz")) == b"1"  # marker present
    assert is_file_processed(db, "a.gz") is True
    assert any("Marked processed: a.gz" in r.getMessage() for r in caplog.records)


def test_get_processed_files_scans_only_prefixed_keys():
    db = FakeShimDB()
    db.put(PROCESSED_PREFIX + b"f1.gz", b"1")
    db.put(b"some/other/key", b"x")
    db.put(PROCESSED_PREFIX + b"f2.gz", b"1")

    got = get_processed_files(db)
    assert got == {"f1.gz", "f2.gz"}


def test_is_file_processed_handles_db_errors(caplog):
    caplog.set_level(logging.WARNING)

    class BadGet(FakeShimDB):
        def get(self, key: bytes):  # type: ignore[override]
            raise RuntimeError("boom")

    db = BadGet()
    assert is_file_processed(db, "x.gz") is False
    assert any("Processed check failed" in r.getMessage() for r in caplog.records)


def test_get_processed_files_handles_db_errors(caplog):
    caplog.set_level(logging.WARNING)

    class BadIter(FakeShimDB):
        def iterator(self):  # type: ignore[override]
            raise RuntimeError("nope")

    db = BadIter()
    assert get_processed_files(db) == set()
    assert any("Could not enumerate processed files" in r.getMessage()
               for r in caplog.records)


def test_get_processed_files_binary_safety_and_stop_on_prefix_exit():
    """Binary keys are bytewise; scan stops exactly when keys no longer share prefix."""
    db = FakeShimDB()
    # in range
    db.put(PROCESSED_PREFIX + b"\x00name.bin", b"1")
    db.put(PROCESSED_PREFIX + b"\xfftail", b"1")
    # out of range (next lexicographic region)
    db.put(b"__processed0__", b"no")         # doesn't share prefix
    db.put(PROCESSED_PREFIX[:-1] + b"0", b"x")  # also no
    got = get_processed_files(db)
    assert "\x00name.bin".encode("utf-8", "ignore").decode("utf-8", "ignore") in {g for g in got}
    assert "tail" in {g for g in got}
    assert "__processed0__" not in got


# -----------------------------------------------------------------------------
# Optional: tiny integration check with the real rocks-shim (auto-skip if absent)
# -----------------------------------------------------------------------------

@pytest.mark.skipif("rocks_shim" not in globals(), reason="rocks_shim import checked below")
def _dummy():  # placeholder to keep pytest happy if globals check is odd
    pass


def _have_real_shim():
    try:
        import rocks_shim as rs  # noqa
        from common_db.api import open_db  # reuse your existing open helper
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_real_shim(), reason="rocks-shim not importable for integration test")
def test_metadata_integration_with_real_shim(tmp_path):
    """
    Sanity check against a real on-disk DB using your open_db helper.
    Skips automatically if rocks_shim/common_db.api aren't importable.
    """
    from common_db.api import open_db

    dbdir = tmp_path / "meta_dbdir"
    dbdir.mkdir(parents=True, exist_ok=True)
    db_path = dbdir / "meta.db"

    with open_db(db_path) as db:
        assert is_file_processed(db, "x.gz") is False
        mark_file_as_processed(db, "x.gz")
        mark_file_as_processed(db, "y.gz")

    # reopen read-only
    with open_db(db_path, mode="ro") as db:
        assert is_file_processed(db, "x.gz") is True
        assert is_file_processed(db, "y.gz") is True
        files = get_processed_files(db)
        assert files == {"x.gz", "y.gz"}
