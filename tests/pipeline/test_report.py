# tests/pipeline/test_report.py
from datetime import datetime

import pytest

from ngram_prep.pipeline.report import (
    format_run_summary,
    log_run_summary,
    print_run_summary,
)


def _demo_kwargs():
    return dict(
        ngram_repo_url="https://example.com/20200217/eng-us/eng-us-2-ngrams_exports.html",
        db_path="/tmp/db",
        file_range=(0, 3),
        file_urls_available=[
            "https://ex/eng-us-2-00001-of-00004.gz",
            "https://ex/eng-us-2-00002-of-00004.gz",
            "https://ex/eng-us-2-00003-of-00004.gz",
            "https://ex/eng-us-2-00004-of-00004.gz",
        ],
        file_urls_to_use=[
            "https://ex/eng-us-2-00001-of-00004.gz",
            "https://ex/eng-us-2-00004-of-00004.gz",
        ],
        ngram_size=2,
        workers=8,
        executor_name="processes",
        start_time=datetime(2025, 8, 18, 12, 34, 56),
        ngram_type="tagged",
        overwrite=True,
        files_to_skip=1,
        write_batch_size=1000,
    )


def test_format_run_summary_no_color_contains_key_fields():
    s = format_run_summary(color=False, **_demo_kwargs())
    # heading formatted
    assert "Start Time: 2025-08-18 12:34:56" in s
    # core fields
    assert "Ngram repository:" in s
    assert "RocksDB database path:      /tmp/db" in s
    assert "File index range:           0 to 3 (count ~ 4)" in s
    assert "Total files available:      4" in s
    assert "Files to process:           2" in s
    assert "Ngram size:                 2" in s
    assert "Ngram filtering:            tagged" in s
    assert "Overwrite mode:             True" in s
    assert "Files to skip (processed):  1" in s
    assert "Write batch size:           1,000" in s
    assert "Worker processes/threads:   8 (processes)" in s
    # no ANSI codes when color=False
    assert "\x1b[" not in s


def test_format_run_summary_color_includes_ansi():
    s = format_run_summary(color=True, **_demo_kwargs())
    assert "\x1b[31m" in s  # red heading
    assert "\x1b[4m" in s   # underlined section title


def test_format_run_summary_truncates_long_urls():
    long_url = "https://example.com/" + ("a" * 200)
    s = format_run_summary(color=False, **_demo_kwargs() | {"ngram_repo_url": long_url})
    # show ellipsis, not the whole tail
    assert "…" in s
    assert long_url not in s  # truncated version printed, not full string


def test_log_run_summary_emits_info(caplog):
    caplog.set_level("INFO")
    log_run_summary(color=False, **_demo_kwargs())
    # at least a couple of expected lines should be present as INFO records
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("Download & Ingestion Configuration" in m for m in messages)
    assert any("RocksDB database path:      /tmp/db" in m for m in messages)


def test_print_run_summary_writes_to_stdout(capfd):
    print_run_summary(color=False, **_demo_kwargs())
    out, _ = capfd.readouterr()
    assert "Download & Ingestion Configuration" in out
    assert "Files to process:           2" in out
