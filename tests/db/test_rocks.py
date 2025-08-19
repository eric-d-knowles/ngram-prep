# tests/db/test_rocks.py
from __future__ import annotations

from pathlib import Path
import logging
import types

import pytest

import ngram_prep.db.rocks as rocks


# --- Fakes ------------------------------------------------------------------ #

class FakeOptions:
    def __init__(self):
        self.create_if_missing_val = False
        self.max_background_jobs = None
        self.write_buffer_size = None
        self.l0_compaction_trigger = None

    # rocksdict-like methods
    def create_if_missing(self, val: bool) -> None:
        self.create_if_missing_val = bool(val)

    def set_max_background_jobs(self, n: int) -> None:
        self.max_background_jobs = int(n)

    def set_write_buffer_size(self, n: int) -> None:
        self.write_buffer_size = int(n)

    def set_level_zero_file_num_compaction_trigger(self, n: int) -> None:
        self.l0_compaction_trigger = int(n)


class FakeRdictFactory:
    """
    Callable that mimics rocksdict.Rdict. You can configure outcomes per call:
    outcomes = [ "lock", "ok", "err" ]
      - "lock": raise Exception with 'lock' in message (triggers retry)
      - "ok":   return a simple object
      - "err":  raise a generic Exception (no retry)
    """
    def __init__(self, outcomes: list[str]):
        self.outcomes = outcomes[:]
        self.calls = 0
        self.last_args = None

    def __call__(self, path: str, opts: FakeOptions):
        self.calls += 1
        self.last_args = (path, opts)
        if not self.outcomes:
            # default to ok if not specified
            return object()

        outcome = self.outcomes.pop(0)
        if outcome == "lock":
            raise Exception("LOCK: resource temporarily unavailable")
        if outcome == "err":
            raise Exception("boom")
        return object()


# --- Tests for make_default_options ----------------------------------------- #

def test_make_default_options_sets_sensible_defaults(monkeypatch):
    monkeypatch.setattr(rocks, "Options", FakeOptions)
    monkeypatch.setattr(rocks.os, "cpu_count", lambda: 2)

    opts = rocks.make_default_options()
    assert isinstance(opts, FakeOptions)
    # create_if_missing set
    assert opts.create_if_missing_val is True
    # background jobs: max(4, cpu_count)
    assert opts.max_background_jobs == 4
    # default buffer and L0 trigger
    assert opts.write_buffer_size == 256 * 1024 * 1024
    assert opts.l0_compaction_trigger == 12


# --- Tests for setup_rocksdb ------------------------------------------------ #

def test_setup_rocksdb_success_creates_parent_and_logs(
    tmp_path: Path, monkeypatch, caplog
):
    monkeypatch.setattr(rocks, "Options", FakeOptions)
    # ensure default options are used
    monkeypatch.setattr(rocks, "make_default_options", lambda: FakeOptions())
    fake_rdict = FakeRdictFactory(["ok"])
    monkeypatch.setattr(rocks, "Rdict", fake_rdict)

    db_path = tmp_path / "db" / "ngram"
    caplog.set_level(logging.INFO)

    obj = rocks.setup_rocksdb(db_path)
    assert obj is not None
    # parent dir must exist
    assert db_path.parent.exists()
    # Rdict called with string path and our FakeOptions
    p_str, opts = fake_rdict.last_args
    assert p_str == str(db_path)
    assert isinstance(opts, FakeOptions)
    # log emitted
    assert any("Opened RocksDB" in rec.getMessage() for rec in caplog.records)


def test_setup_rocksdb_retries_on_lock_then_succeeds(
    tmp_path: Path, monkeypatch, caplog
):
    monkeypatch.setattr(rocks, "Options", FakeOptions)
    monkeypatch.setattr(rocks, "make_default_options", lambda: FakeOptions())

    # first call: lock error, second call: ok
    fake_rdict = FakeRdictFactory(["lock", "ok"])
    monkeypatch.setattr(rocks, "Rdict", fake_rdict)

    sleeps: list[float] = []
    monkeypatch.setattr(rocks.time, "sleep", lambda s: sleeps.append(s))

    caplog.set_level(logging.WARNING)
    db_path = tmp_path / "db" / "ngram"

    obj = rocks.setup_rocksdb(
        db_path, retries=2, delay_seconds=0.1, backoff=3.0
    )
    assert obj is not None
    # one retry -> one sleep recorded
    assert sleeps == [0.1]
    assert fake_rdict.calls == 2
    # warn
