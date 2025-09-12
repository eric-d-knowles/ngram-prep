# tests/pipeline/test_report.py
from __future__ import annotations

import logging
from datetime import datetime

import pytest

from ngram_acquire.pipeline.report import (
    _abbrev,
    format_run_summary,
    print_run_summary,
    log_run_summary,
)

RED = "\033[31m"
UNDER = "\033[4m"
RESET = "\033[0m"


# -----------------
# _abbrev behavior
# -----------------

def test_abbrev_no_truncation():
    s = "hello"
    assert _abbrev(s, width=5) == "hello"
    assert _abbrev(s, width=6) == "hello"


def test_abbrev_truncates_and_uses_ellipsis():
    s = "abcdefghij"
    out = _abbrev(s, width=5)
    assert out.endswith("…")
    assert out == "abcd…"
    # width=1 -> only ellipsis
    assert _abbrev("abc", width=1) == "…"
    # width=0 -> also ellipsis (due to max(0, width-1))
    assert _abbrev("abc", width=0) == "…"


# ----------------------------
# format_run_summary behavior
# ----------------------------

BASE_KW = dict(
    ngram_repo_url="https://example.com/ngrams/really/long/path/filelist.json",
    db_path="/var/data/rocks/db",
    file_range=(10, 19),
    file_urls_available=[f"https://host/f{i:02d}.gz" for i in range(25)],
    file_urls_to_use=[f"https://host/f{i:02d}.gz" for i in range(10, 20)],
    ngram_size=5,
    workers=8,
    executor_name="process",
    start_time=datetime(2024, 1, 2, 3, 4, 5),
    ngram_type="all",
    overwrite=True,
    files_to_skip=0,
    write_batch_size=0,
)


def test_format_run_summary_plain_text_basics():
    s = format_run_summary(color=False, **BASE_KW)
    # Heading
    assert "Start Time: 2024-01-02 03:04:05" in s
    # Section title not colored/underlined when color=False
    assert "Download & Ingestion Configuration" in s
    assert UNDER not in s and RED not in s

    # Range count is inclusive: 10..19 -> 10 files
    assert "File index range:           10 to 19 (count ~ 10)" in s

    # Totals
    assert "Total files available:      25" in s
    assert "Files to process:           10" in s

    # First/last URLs (not abbreviated here)
    assert "First file URL:             https://host/f10.gz" in s
    assert "Last file URL:              https://host/f19.gz" in s

    # Core params
    assert "Ngram size:                 5" in s
    assert "Ngram filtering:            all" in s
    assert "Overwrite mode:             True" in s
    assert "Worker processes/threads:   8 (process)" in s
    # Ends with newline
    assert s.endswith("\n")


def test_format_run_summary_with_color_and_extras():
    kw = dict(BASE_KW)
    kw.update(files_to_skip=3, write_batch_size=1234567, color=True)
    s = format_run_summary(**kw)

    # Colored heading & underlined section header
    assert f"{RED}Start Time: 2024-01-02 03:04:05{RESET}" in s
    assert f"{UNDER}\nDownload & Ingestion Configuration{RESET}" in s

    # Extra lines present
    assert "Files to skip (processed):  3" in s
    assert "Write batch size:           1,234,567" in s  # comma formatted


def test_format_run_summary_empty_urls_and_negative_range():
    kw = dict(BASE_KW)
    kw.update(file_range=(5, 3), file_urls_to_use=[], file_urls_available=[])
    s = format_run_summary(color=False, **kw)
    # count clamps at 0
    assert "File index range:           5 to 3 (count ~ 0)" in s
    # First/last safely handled
    assert "First file URL:             None" in s
    assert "Last file URL:              None" in s
    # Totals are zero
    assert "Total files available:      0" in s
    assert "Files to process:           0" in s


def test_format_run_summary_abbreviates_long_repo_url():
    kw = dict(BASE_KW)
    # Create a very long URL
    kw["ngram_repo_url"] = "https://" + "a" * 200
    s = format_run_summary(color=False, **kw)
    # Should contain an ellipsis on the repo line
    assert "Ngram repository:" in s
    assert "…" in s.split("Ngram repository:")[1]


# ----------------------------
# print_run_summary & logging
# ----------------------------

def test_print_run_summary_captures_stdout(capsys):
    print_run_summary(color=False, **BASE_KW)
    out = capsys.readouterr().out
    assert "Download & Ingestion Configuration" in out
    assert out.endswith("\n")


def test_log_run_summary_emits_info_lines(caplog):
    caplog.set_level(logging.INFO)
    log_run_summary(color=False, **BASE_KW)

    # Each line should be logged at INFO, without trailing newline
    msgs = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.INFO]
    assert any(m.startswith("Start Time: 2024-01-02 03:04:05") for m in msgs)
    assert any("Download & Ingestion Configuration" in m for m in msgs)
    assert any("Worker processes/threads:   8 (process)" in m for m in msgs)
