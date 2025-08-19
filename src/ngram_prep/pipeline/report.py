# ngram_prep/pipeline/report.py
from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence, Tuple

logger = logging.getLogger(__name__)


def _abbrev(s: str, width: int = 96) -> str:
    """Return s truncated with an ellipsis if it exceeds width."""
    return s if len(s) <= width else (s[: max(0, width - 1)] + "…")


def format_run_summary(
    *,
    ngram_repo_url: str,
    db_path: str,
    file_range: Tuple[int, int],
    file_urls_available: Sequence[str],
    file_urls_to_use: Sequence[str],
    ngram_size: int,
    workers: int,
    executor_name: str,
    start_time: datetime,
    ngram_type: str = "all",
    overwrite: bool = True,
    files_to_skip: int = 0,
    write_batch_size: int = 0,  # pass your DEFAULT_WRITE_BATCH_SIZE here
    color: bool = True,
) -> str:
    """
    Build a formatted, human-readable summary of the planned run.
    """
    heading = f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}"
    if color:
        heading = f"\033[31m{heading}\033[0m"

    start, end = file_range
    count_in_range = max(0, end - start + 1)

    first_url = file_urls_to_use[0] if file_urls_to_use else "None"
    last_url = file_urls_to_use[-1] if file_urls_to_use else "None"

    lines = [
        heading,
        ("\033[4mDownload & Ingestion Configuration\033[0m" if color
         else "Download & Ingestion Configuration"),
        f"Ngram repository:           {_abbrev(ngram_repo_url)}",
        f"RocksDB database path:      {db_path}",
        f"File index range:           {start} to {end} "
        f"(count ~ {count_in_range})",
        f"Total files available:      {len(file_urls_available)}",
        f"Files to process:           {len(file_urls_to_use)}",
        f"First file URL:             {_abbrev(first_url)}",
        f"Last file URL:              {_abbrev(last_url)}",
        f"Ngram size:                 {ngram_size}",
        f"Ngram filtering:            {ngram_type}",
        f"Overwrite mode:             {overwrite}",
    ]

    if files_to_skip > 0:
        lines.append(f"Files to skip (processed):  {files_to_skip}")

    if write_batch_size:
        lines.append(f"Write batch size:           {write_batch_size:,}")

    lines.append(f"Worker processes/threads:   {workers} ({executor_name})")
    return "\n".join(lines) + "\n"


def print_run_summary(**kwargs) -> None:
    """Print the run summary to stdout (CLI usage)."""
    print(format_run_summary(**kwargs), end="")


def log_run_summary(*, color: bool = False, **kwargs) -> None:
    """Log the run summary at INFO level (pipelines using logging)."""
    summary = format_run_summary(color=color, **kwargs)
    for line in summary.rstrip("\n").splitlines():
        logger.info(line)
