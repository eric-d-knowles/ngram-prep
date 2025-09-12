# tests/ngram_filter/pipeline/test_progress.py
from __future__ import annotations

import multiprocessing as mp
import time

import ngram_filter.pipeline.progress as mod


def _ctx():
    try:
        return mp.get_context("fork")
    except (ValueError, RuntimeError):
        return mp.get_context("spawn")


def test_make_counters_inc_and_read():
    ctx = _ctx()
    c = mod.make_counters(ctx)

    # All zeros initially
    assert mod._read(c.items_scanned) == 0
    assert mod._read(c.items_filtered) == 0
    assert mod._read(c.items_enqueued) == 0
    assert mod._read(c.reader_done) == 0
    assert mod._read(c.items_buffered) == 0
    assert mod._read(c.items_written) == 0
    assert mod._read(c.bytes_flushed) == 0
    assert mod._read(c.opened_dbs) == 0
    assert mod._read(c.closed_dbs) == 0

    # Increment a few and verify
    mod._inc(c.items_scanned, 5)
    mod._inc(c.items_filtered, 2)
    mod._inc(c.items_enqueued, 3)
    mod._inc(c.reader_done, 1)
    mod._inc(c.items_written, 7)
    mod._inc(c.opened_dbs, 1)
    mod._inc(c.closed_dbs, 1)

    assert mod._read(c.items_scanned) == 5
    assert mod._read(c.items_filtered) == 2
    assert mod._read(c.items_enqueued) == 3
    assert mod._read(c.reader_done) == 1
    assert mod._read(c.items_written) == 7
    assert mod._read(c.opened_dbs) == 1
    assert mod._read(c.closed_dbs) == 1


def test_snapshot_and_format_stats_contains_expected_fields():
    ctx = _ctx()
    c = mod.make_counters(ctx)

    # Bump a representative set
    mod._inc(c.items_scanned, 10)
    mod._inc(c.items_filtered, 4)
    mod._inc(c.items_enqueued, 6)
    mod._inc(c.reader_done, 2)
    mod._inc(c.items_buffered, 6)
    mod._inc(c.items_written, 6)
    mod._inc(c.bytes_flushed, 1234)
    mod._inc(c.opened_dbs, 2)
    mod._inc(c.closed_dbs, 2)

    snap = mod.snapshot_counters(c)
    assert snap["items_scanned"] == 10
    assert snap["items_filtered"] == 4
    assert snap["items_enqueued"] == 6
    assert snap["reader_done"] == 2
    assert snap["items_buffered"] == 6
    assert snap["items_written"] == 6
    assert snap["bytes_flushed"] == 1234
    assert snap["opened_dbs"] == 2
    assert snap["closed_dbs"] == 2

    # Use a start time ~1s in the past so rate is finite and string stable-ish
    start_time = time.perf_counter() - 1.0
    s = mod.format_stats(c, start_time)

    # Check for key fields; don't overfit on exact formatting of numbers
    for needle in [
        "Scanned: 10",
        "Filtered: 4",
        "Enqueued: 6",
        "Readers Done: 2",
        "Buffered: 6",
        "Written: 6",
        "Flushed: 1,234 B" if "," in s else "Flushed: 1234 B",  # tolerate locale/formatting
        "DBs Open/Close: 2/2",
        "Rate:",
        "Elapsed:",
    ]:
        assert needle in s


def test_reporter_prints_final_line_when_stopped_immediately(capsys):
    ctx = _ctx()
    c = mod.make_counters(ctx)
    mod._inc(c.items_scanned, 5)

    start_time = time.perf_counter() - 0.5
    stop_flag = ctx.Value("b", 1)  # already set

    mod.reporter(c, start_time, every_s=10.0, stop_flag=stop_flag)

    out = capsys.readouterr().out
    lines = [ln for ln in out.strip().splitlines() if ln]

    # Contains the key fields…
    assert any("Scanned: 5" in ln for ln in lines)
    assert any("Elapsed:" in ln for ln in lines)

    # …and we only printed once (2–3 lines depending on format_stats layout).
    assert 1 <= len(lines) <= 3

