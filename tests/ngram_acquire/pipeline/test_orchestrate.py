# tests/pipeline/test_orchestrate.py
from __future__ import annotations

import re
from pathlib import PurePosixPath

import pytest

from ngram_acquire.pipeline.orchestrate import download_and_ingest_to_rocksdb


class StubDB:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True  # not strictly needed; context will close


def _install_stubs(monkeypatch, *, urls):
    """
    Install minimal stubs on the orchestrator module so the function
    runs without network or RocksDB. Returns a dict of call-capture refs.
    """
    import ngram_acquire.pipeline.orchestrate as orch

    calls = {
        "cleanup": [],
        "open_db": [],        # (path, profile)
        "process_files": [],
        "print_summary": [],
        "is_processed": [],
        "shuffle_called": False,
    }

    # set_location_info
    monkeypatch.setattr(
        orch,
        "set_location_info",
        lambda n, rel, corp: ("https://ex/page.html", re.compile(r"^eng-us-1-\d+\.gz$")),
        raising=True,
    )

    # fetch_file_urls
    monkeypatch.setattr(orch, "fetch_file_urls", lambda *_: list(urls), raising=True)

    # make_ngram_type_predicate
    monkeypatch.setattr(orch, "make_ngram_type_predicate", lambda *_: (lambda s: True), raising=True)

    # print_run_summary: capture call
    def _print_summary(**kwargs):
        calls["print_summary"].append(kwargs)
    monkeypatch.setattr(orch, "print_run_summary", _print_summary, raising=True)

    # safe_db_cleanup
    def _cleanup(path):
        calls["cleanup"].append(path)
        return True
    monkeypatch.setattr(orch, "safe_db_cleanup", _cleanup, raising=True)

    # open_db: context manager yielding StubDB and capturing path/profile
    class _OpenCtx:
        def __init__(self, path, *, profile=None, **_k):
            calls["open_db"].append((str(path), profile))
            self._db = StubDB()
        def __enter__(self):
            return self._db
        def __exit__(self, exc_type, exc, tb):
            self._db.closed = True
            return False

    monkeypatch.setattr(orch, "open_db", lambda path, **k: _OpenCtx(path, **k), raising=True)

    # is_file_processed: consult injected set
    def _is_processed(db, name):
        calls["is_processed"].append(name)
        proc = getattr(orch, "_TEST_PROCESSED", set())
        return name in proc
    monkeypatch.setattr(orch, "is_file_processed", _is_processed, raising=True)

    # process_files: capture filenames; return synthetic stats
    def _process_files(*, urls, executor_class, workers, db, filter_pred, write_batch_size):
        names = [PurePosixPath(u).name for u in urls]
        calls["process_files"].append(dict(names=names, workers=workers, exec=executor_class))
        success = [f"SUCCESS: {names[0]}", f"SUCCESS: {names[-1]}"] if names else []
        failure = []
        return success, failure, 123, 2
    monkeypatch.setattr(orch, "process_files", _process_files, raising=True)

    # Optional log monitor
    if getattr(orch, "start_rocksdb_log_monitor", None) is not None:
        monkeypatch.setattr(orch, "start_rocksdb_log_monitor", lambda *a, **k: None, raising=True)

    # random.shuffle hook
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
        open_type="default",
    )

    # Clea
