"""Run summary reporting for ngram acquisition pipeline."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = ["format_run_summary", "print_run_summary", "log_run_summary"]


def _abbrev(s: str, width: int = 96) -> str:
    """Truncate string with ellipsis if it exceeds width."""
    return s if len(s) <= width else s[: width - 1] + "…"


def format_run_summary(
        *,
        ngram_repo_url: str,
        db_path: str,
        file_range: Tuple[int, int],
        file_urls_available: Sequence[str],
        file_urls_to_use: Sequence[str],
        ngram_size: int,
        workers: int,
        start_time: datetime,
        ngram_type: str = "all",
        overwrite_db: bool = False,
        overwrite_checkpoint: bool = False,
        files_to_skip: int = 0,
        write_batch_size: int = 0,
        post_compact: bool = False,
        prefix_bytes: int = None,
        profile: str = "read:packed24",
) -> str:
    """
    Build a formatted summary of the planned pipeline run.

    Args:
        ngram_repo_url: Repository URL being accessed
        db_path: Path to RocksDB database
        file_range: (start_index, end_index) of files to process
        file_urls_available: All available file URLs from repository
        file_urls_to_use: Subset of URLs that will be processed
        ngram_size: N-gram size (1-5)
        workers: Number of worker processes/threads
        executor_name: "processes" or "threads"
        start_time: Pipeline start timestamp
        ngram_type: Filter type ("all", "tagged", etc.)
        overwrite_db: Whether to overwrite existing database
        overwrite_checkpoint: Whether to overwrite existing checkpoint
        files_to_skip: Number of already-processed files being skipped
        write_batch_size: Number of entries per batch write
        post_compact: Whether post-ingestion compaction is enabled
        prefix_bytes: Prefix bytes for compaction ranges
        profile: RocksDB configuration profile name

    Returns:
        Formatted summary string with newline at end
    """
    start, end = file_range

    lines = [
        "N-GRAM ACQUISITION PIPELINE",
        "━" * 100,
        f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}",
        "",
        "Download Configuration",
        "═" * 100,
        f"Ngram repo:           {_abbrev(ngram_repo_url)}",
        f"DB path:              {db_path}",
        f"File range:           {start} to {end}",
        f"Total files:          {len(file_urls_available)}",
        f"Files to get:         {len(file_urls_to_use)}",
        f"Skipping:             {files_to_skip}",
        f"Download workers:     {workers}",
        f"Batch size:           {write_batch_size:,}",
        f"Ngram size:           {ngram_size}",
        f"Ngram type:           {ngram_type}",
        f"Overwrite DB:         {overwrite_db}",
        f"Overwrite checkpoint: {overwrite_checkpoint}",
        f"DB Profile:           {profile}",
        f"Compact:              {post_compact}",
        "\nDownload Progress",
        "═" * 100,
    ]

    return "\n".join(lines) + "\n"


def print_run_summary(**kwargs) -> None:
    """
    Print the run summary to stdout.

    Args:
        **kwargs: All arguments accepted by format_run_summary()
    """
    print(format_run_summary(**kwargs), end="")


def log_run_summary(**kwargs) -> None:
    """
    Log the run summary at INFO level.

    Logs each line separately for better log file readability.

    Args:
        **kwargs: All arguments accepted by format_run_summary()
    """
    summary = format_run_summary(**kwargs)
    for line in summary.rstrip("\n").splitlines():
        logger.info(line)