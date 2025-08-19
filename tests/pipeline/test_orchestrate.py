# tests/pipeline/test_orchestrate.py
from __future__ import annotations

import re
from pathlib import PurePosixPath

import pytest

# Adjust this import if you placed the function elsewhere
from ngram_prep.pipeline.orchestrate import download_and_ingest_to_rocksdb


class StubDB:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _install_stubs(monkeypatch, *, urls):
    """
    Install minimal stubs on the orchestrator module so the function
    runs without network or RocksDB. Returns a dict of call-capture refs.
    """
    import ngram_prep.pipeline.orchestrate as orch

    calls = {
        "cleanup": [],
        "setup_db": [],
        "process_files": [],
        "print_summary": [],
        "is_processed": [],
        "shuffle_called": False,
    }

    # set_location_info: return page URL + compiled pattern (ignored by stub)
    monkeypatch.setattr(
        orch, "set_location_info",
        lambda n, rel, corp: ("https://ex/page.html",
                              re.compile(r"^eng-us-1-\d+\.gz$")),
        raising=True,
    )

    # fetch_file_urls: return our fake list
    monkeypatch.setattr(orch, "fetch_file_urls", lambda *_: list(urls), raising=True)

    # make_ngram_type_predicate: accept everything
    monkeypatch.setattr(
        orch, "make_ngram_type_predicate", lambda *_: (lambda s: True), raising=True
    )

    # print_run_summary: capture call, no output
    def _print_summary(**kwargs):
        calls["print_summary"].append(kwargs)
    monkeypatch.setattr(orch, "print_run_summary", _print_summary, raising=True)

    # safe_db_cleanup: record call, say success
    def _cleanup(path):
        calls["cleanup"].append(path)
        return True
    monkeypatch.setattr(orch, "safe_db_cleanup", _cleanup, raising=True)

    # setup_rocksdb: return stub db; capture path
    def _setup_db(path):
        calls["setup_db"].append(path)
        return StubDB()
    monkeypatch.setattr(orch, "setup_rocksdb", _setup_db, raising=True)

    # is_file_processed: consult an injected set on the module
    def _is_processed(db, name):
        calls["is_processed"].append(name)
        proc = getattr(orch, "_TEST_PROCESSED", set())
        return name in proc
    monkeypatch.setattr(orch, "is_file_processed", _is_processed, raising=True)

    # process_files: capture urls passed in; return synthetic stats
    def _process_files(*, urls, executor_class, workers, db,
                       filter_pred, write_batch_size):
        # Record just the filenames
        names = [PurePosixPath(u).name for u in urls]
        calls["process_files"].append(
            dict(names=names, workers=workers, exec=executor_class)
        )
        # Simulate two successes, one failure, and some counts
        success = [f"SUCCESS: {names[0]}", f"SUCCESS: {names[-1]}"] if names else []
        failure = []
        return success, failure, 123, 2
    monkeypatch.setattr(orch, "process_files", _process_files, raising=True)

    # Optional log monitor: make it a no-op if present
    if getattr(orch, "start_rocksdb_log_monitor", None) is not None:
        monkeypatch.setattr(
            orch, "start_rocksdb_log_monitor", lambda *a, **k: None, raising=True
        )

    # Make random.shuffle visible in calls (used only when random_seed is set)
    import random as _random

    def _shuffle(seq):
        calls["shuffle_called"] = True
        seq.reverse()  # deterministic transform for assertion
    monkeypatch.setattr(orch.random, "shuffle", _shuffle, raising=True)

    return calls


def test_overwrite_calls_cleanup_and_runs_process(tmp_path, monkeypatch, capsys):
    # Create an existing DB dir to trigger cleanup
    db_path = tmp_path / "dbdir"
    db_path.mkdir()

    urls = [
        "https://ex/eng-us-1-00000-of-00002.gz",
        "https://ex/eng-us-1-00001-of-00002.gz",
    ]
    calls = _install_stubs(monkeypatch, urls=urls)

    download_and_ingest_to_rocksdb(
        ngram_size=1,
        repo_release_id="20200217",
        repo_corpus_id="eng-us-all",
        db_path=str(db_path),
        overwrite=True,
        workers=2,
        use_threads=True,
        ngram_type="all",
        write_batch_size=100,
    )

    # Cleanup was invoked on the DB path
    assert calls["cleanup"] == [str(db_path)]
    # DB opened exactly once
    assert calls["setup_db"] == [str(db_path)]
    # Runner was called once with both files (filenames)
    assert calls["process_files"] and calls["process_files"][0]["names"] == [
        "eng-us-1-00000-of-00002.gz",
        "eng-us-1-00001-of-00002.gz",
    ]
    # Human summary printed (via our stub)
    assert calls["print_summary"]

    # Completion banner printed to stdout
    out, _ = capsys.readouterr()
    assert "Processing completed!" in out


def test_resume_skips_processed_and_early_exits(monkeypatch, capsys):
    import ngram_prep.pipeline.orchestrate as orch

    urls = [
        "https://ex/a.gz",
        "https://ex/b.gz",
    ]
    calls = _install_stubs(monkeypatch, urls=urls)
    # Pretend both are already processed
    orch._TEST_PROCESSED = {"a.gz", "b.gz"}  # type: ignore[attr-defined]

    download_and_ingest_to_rocksdb(
        ngram_size=1,
        repo_release_id="20200217",
        repo_corpus_id="eng-us-all",
        db_path="/tmp/whatever",
        overwrite=False,  # resume mode
        workers=2,
    )

    # process_files should NOT be called when everything is already processed
    assert calls["process_files"] == []
    out, _ = capsys.readouterr()
    assert "already processed" in out


def test_invalid_range_raises(monkeypatch):
    urls = [
        "https://ex/f0.gz",
        "https://ex/f1.gz",
    ]
    _install_stubs(monkeypatch, urls=urls)

    with pytest.raises(ValueError):
        download_and_ingest_to_rocksdb(
            ngram_size=1,
            repo_release_id="20200217",
            repo_corpus_id="eng-us-all",
            db_path="/tmp/db",
            file_range=(0, 5),  # invalid (end out of bounds)
        )


def test_random_seed_triggers_shuffle(monkeypatch):
    urls = [
        "https://ex/f0.gz",
        "https://ex/f1.gz",
        "https://ex/f2.gz",
    ]
    calls = _install_stubs(monkeypatch, urls=urls)

    download_and_ingest_to_rocksdb(
        ngram_size=1,
        repo_release_id="20200217",
        repo_corpus_id="eng-us-all",
        db_path="/tmp/db",
        overwrite=True,
        workers=2,
        random_seed=42,  # triggers orchestrator's shuffle path
    )

    assert calls["shuffle_called"] is True
    # Because our stub shuffle reverses in-place, ensure that reversal reached runner
    seen = calls["process_files"][0]["names"]
    assert seen == ["f2.gz", "f1.gz", "f0.gz"]
