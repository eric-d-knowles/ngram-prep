# tests/pipeline/test_worker.py
from __future__ import annotations

import gzip
import io
import struct
from pathlib import PurePosixPath

import pytest
import requests

from ngram_acquire.pipeline.worker import process_and_ingest_file
from ngram_acquire.utils.filters import make_ngram_type_predicate


# --- helpers -----------------------------------------------------------------


def _gz_bytes(lines: list[bytes]) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(b"\n".join(lines))
    return buf.getvalue()


class FakeResponse:
    def __init__(self, gz_payload: bytes, headers: dict[str, str] | None = None):
        self.raw = io.BytesIO(gz_payload)
        self.headers = headers or {}
        self._closed = False

    def close(self) -> None:
        self._closed = True


# --- tests -------------------------------------------------------------------


def test_success_parses_and_packs(monkeypatch):
    # 3 lines: 2 tagged (kept), 1 untagged (filtered)
    lines = [
        b"alpha_NOUN\t2000,10,2\t2001,20,4",
        b"beta_VERB\t1999,3,5",
        b"gamma delta\t2001,1,1",
    ]
    resp = FakeResponse(_gz_bytes(lines), headers={"content-length": "1234"})

    # Patch download to return our fake streaming response
    import ngram_acquire.pipeline.worker as worker_mod
    monkeypatch.setattr(worker_mod, "stream_download_with_retries", lambda *a, **k: resp)

    pred = make_ngram_type_predicate("tagged")

    msg, data = process_and_ingest_file(
        "https://example.com/eng-us-1-00000-of-00001.gz",
        worker_id=1,
        filter_pred=pred,
    )

    # Status and filename presence
    assert msg.startswith("SUCCESS:")
    assert "3 lines" in msg
    assert PurePosixPath("eng-us-1-00000-of-00001.gz").name in msg

    # Only tagged keys kept
    assert set(data.keys()) == {"alpha_NOUN", "beta_VERB"}

    # alpha has 2 year tuples -> 6 uint64s -> 48 bytes
    assert len(data["alpha_NOUN"]) == 6 * 8
    # beta has 1 year tuple -> 3 uint64s -> 24 bytes
    assert len(data["beta_VERB"]) == 3 * 8

    # Optional: verify exact values for one key
    vals = struct.unpack("<6Q", data["alpha_NOUN"])
    assert vals == (2000, 10, 2, 2001, 20, 4)


def test_unicode_decode_error_is_logged_and_processing_continues(monkeypatch, caplog):
    lines = [
        b"x_NOUN\t1990,1,1",
        b"\xff\xfe",  # invalid UTF-8
        b"y_VERB\t1991,2,2",
    ]
    resp = FakeResponse(_gz_bytes(lines), headers={"content-length": "weird"})

    import ngram_acquire.pipeline.worker as worker_mod
    monkeypatch.setattr(worker_mod, "stream_download_with_retries", lambda *a, **k: resp)

    pred = make_ngram_type_predicate("tagged")
    caplog.set_level("WARNING")

    msg, data = process_and_ingest_file(
        "https://ex/file.gz", worker_id=2, filter_pred=pred
    )

    # two valid tagged entries packed despite one bad line
    assert set(data.keys()) == {"x_NOUN", "y_VERB"}
    assert "3 lines" in msg
    # saw a Unicode warning
    assert any("Unicode error" in r.getMessage() for r in caplog.records)


def test_timeout_returns_message_and_empty_dict(monkeypatch):
    import ngram_acquire.pipeline.worker as worker_mod

    def boom(*a, **k):
        raise requests.Timeout("slow")

    monkeypatch.setattr(worker_mod, "stream_download_with_retries", boom)

    msg, data = process_and_ingest_file("https://ex/file.gz", worker_id=3)
    assert msg.startswith("TIMEOUT:")
    assert data == {}


def test_network_error_returns_message_and_empty_dict(monkeypatch):
    import ngram_acquire.pipeline.worker as worker_mod

    def boom(*a, **k):
        raise requests.RequestException("down")

    monkeypatch.setattr(worker_mod, "stream_download_with_retries", boom)

    msg, data = process_and_ingest_file("https://ex/file.gz", worker_id=4)
    assert msg.startswith("NETWORK_ERROR:")
    assert data == {}


def test_generic_exception_returns_message_and_empty_dict(monkeypatch):
    import ngram_acquire.pipeline.worker as worker_mod

    def boom(*a, **k):
        raise RuntimeError("oops")

    monkeypatch.setattr(worker_mod, "stream_download_with_retries", boom)

    msg, data = process_and_ingest_file("https://ex/file.gz", worker_id=5)
    assert msg.startswith("ERROR:")
    assert data == {}
