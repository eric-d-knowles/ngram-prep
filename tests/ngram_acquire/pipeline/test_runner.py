# tests/pipeline/test_runner.py
from __future__ import annotations

from concurrent.futures import Future
from pathlib import PurePosixPath

import pytest

from ngram_acquire.pipeline.runner import process_files


# --- Test doubles -------------------------------------------------------------

class Events:
    def __init__(self):
        self.order: list[tuple[str, str | list[str]]] = []
        self.writes: list[int] = []
        self.marks: list[str] = []
        self.submitted: list[tuple[str, int]] = []
        self.executor_workers: list[int] = []
        self.fail_db_write: bool = False
        self.results_by_url: dict[str, tuple[str, dict[str, bytes]]] = {}


class FakeExecutor:
    """Minimal context-manager executor with immediate execution."""
    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self._events: Events | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        # Run immediately; wrap result in Future so as_completed works
        fut = Future()
        try:
            res = fn(*args, **kwargs)
            fut.set_result(res)
        except Exception as e:  # pragma: no cover (not used here)
            fut.set_exception(e)
        return fut


class DummyTqdm:
    """No-op tqdm stand-in to keep tests quiet."""
    def __init__(self, total=0, desc="", unit="", colour=None):
        self.total = total

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def update(self, *_):
        pass


# --- Fixture to install stubs on runner module --------------------------------

@pytest.fixture
def install_runner_stubs(monkeypatch):
    """
    Install stubs on the runner module and return an Events collector
    plus the FakeExecutor class to pass as executor_class.
    """
    import ngram_acquire.pipeline.runner as runner_mod

    ev = Events()

    # Stub tqdm
    monkeypatch.setattr(runner_mod, "tqdm", DummyTqdm, raising=True)

    # Stub process_and_ingest_file: pull result from ev.results_by_url
    def _worker(url, worker_id, filter_pred, *, session=None):
        # record the submission (filename, worker_id)
        name = PurePosixPath(url).name
        ev.submitted.append((name, worker_id))
        return ev.results_by_url[url]
    monkeypatch.setattr(runner_mod, "process_and_ingest_file", _worker, raising=True)

    # Stub write_batch_to_db: record writes, optionally raise
    def _write(db, pending_data):
        if ev.fail_db_write:
            raise RuntimeError("db down")
        ev.writes.append(len(pending_data))
        ev.order.append(("write", []))
        return len(pending_data)
    monkeypatch.setattr(runner_mod, "write_batch_to_db", _write, raising=True)

    # Stub mark_file_as_processed: record marks and order
    def _mark(db, filename):
        ev.marks.append(filename)
        ev.order.append(("mark", filename))
    monkeypatch.setattr(runner_mod, "mark_file_as_processed", _mark, raising=True)

    # Wrap FakeExecutor so we can capture workers passed in
    class CapturingExecutor(FakeExecutor):
        def __init__(self, max_workers: int):
            super().__init__(max_workers)
            ev.executor_workers.append(max_workers)

    return ev, CapturingExecutor


# --- Tests --------------------------------------------------------------------


def test_flush_and_mark_after_write(install_runner_stubs):
    ev, CapturingExecutor = install_runner_stubs

    urls = [
        "https://ex/a.gz",
        "https://ex/b.gz",
    ]
    # Each worker returns one entry -> batch size 2 triggers a single flush
    ev.results_by_url = {
        urls[0]: ("SUCCESS: a.gz", {"a": b"x"}),
        urls[1]: ("SUCCESS: b.gz", {"b": b"y"}),
    }

    success, failure, written, batches = process_files(
        urls=urls,
        executor_class=CapturingExecutor,
        workers=3,
        db=object(),  # not used by our stubs
        filter_pred=None,
        write_batch_size=2,
    )

    # Both successes, no failures
    assert len(success) == 2
    assert failure == []

    # Exactly one write with 2 entries
    assert ev.writes == [2]
    assert written == 2
    assert batches == 1

    # Mark happens only after write (order: write then marks)
    assert ev.order and ev.order[0][0] == "write"
    # Two marks, for both filenames
    assert set(ev.marks) == {"a.gz", "b.gz"}

    # Executor saw the requested workers
    assert ev.executor_workers == [3]


def test_failure_is_not_marked_and_is_reported(install_runner_stubs):
    ev, CapturingExecutor = install_runner_stubs

    urls = [
        "https://ex/ok.gz",
        "https://ex/bad.gz",
    ]
    ev.results_by_url = {
        urls[0]: ("SUCCESS: ok.gz", {"ok": b"1"}),
        urls[1]: ("ERROR: bad.gz", {}),
    }

    success, failure, written, batches = process_files(
        urls=urls,
        executor_class=CapturingExecutor,
        workers=1,
        db=object(),
        filter_pred=None,
        write_batch_size=1,  # flush after the success
    )

    # One success, one failure
    assert len(success) == 1
    assert len(failure) == 1
    assert "ERROR: bad.gz" in failure[0]

    # Exactly one write, one mark (for ok.gz only)
    assert ev.writes == [1]
    assert ev.marks == ["ok.gz"]
    assert written == 1
    assert batches == 1


def test_db_write_error_records_failures_and_does_not_mark(install_runner_stubs):
    ev, CapturingExecutor = install_runner_stubs
    ev.fail_db_write = True

    urls = ["https://ex/e.gz"]
    ev.results_by_url = {
        urls[0]: ("SUCCESS: e.gz", {"e": b"!!"}),
    }

    success, failure, written, batches = process_files(
        urls=urls,
        executor_class=CapturingExecutor,
        workers=1,
        db=object(),
        filter_pred=None,
        write_batch_size=1,
    )

    # write failed -> no marks, failure message emitted, counts zeroed
    assert ev.marks == []
    assert any("DB_WRITE_ERROR" in msg for msg in failure)
    assert written == 0  # our runner increments only on successful write
    assert batches == 0


def test_final_flush_when_threshold_not_reached(install_runner_stubs):
    ev, CapturingExecutor = install_runner_stubs

    urls = ["https://ex/u.gz", "https://ex/v.gz"]
    ev.results_by_url = {
        urls[0]: ("SUCCESS: u.gz", {"u": b"x"}),
        urls[1]: ("SUCCESS: v.gz", {"v": b"y"}),
    }

    # Threshold too high -> no mid-run flush; final flush at the end
    success, failure, written, batches = process_files(
        urls=urls,
        executor_class=CapturingExecutor,
        workers=2,
        db=object(),
        filter_pred=None,
        write_batch_size=10,
    )

    assert len(success) == 2 and not failure
    assert ev.writes == [2]  # one final flush
    assert set(ev.marks) == {"u.gz", "v.gz"}
    assert written == 2
    assert batches == 1
