"""Statistics and display reporting for acquisition pipeline."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from ngramprep.utilities.display import format_bytes, format_banner, truncate_path_to_fit

__all__ = ["print_pipeline_header", "print_final_summary"]


def print_pipeline_header(
    start_time: datetime,
    page_url: str,
    db_path: str,
    start_idx: int,
    end_idx: int,
    total_files: int,
    files_to_get: int,
    files_to_skip: int,
    workers: int,
    write_batch_size: int,
    ngram_size: int,
    ngram_type: str,
    overwrite_db: bool,
    open_type: str,
) -> None:
    """
    Print pipeline configuration header.

    Args:
        start_time: Pipeline start timestamp
        page_url: Repository URL
        db_path: Database path
        start_idx: Starting file index
        end_idx: Ending file index
        total_files: Total files available
        files_to_get: Number of files to process
        files_to_skip: Number of files being skipped (resume mode)
        workers: Number of worker processes
        write_batch_size: Batch size for writes
        ngram_size: N-gram size
        ngram_type: Type filter (all/tagged/untagged)
        overwrite_db: Whether overwriting database
        open_type: RocksDB profile name
    """
    print(format_banner("N-GRAM ACQUISITION PIPELINE", style="━"))
    print(f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}")
    print()
    print(format_banner("Download Configuration"))
    print(f"Ngram repo:           {truncate_path_to_fit(page_url, 'Ngram repo:           ')}")
    print(f"DB path:              {truncate_path_to_fit(db_path, 'DB path:              ')}")
    print(f"File range:           {start_idx} to {end_idx}")
    print(f"Total files:          {total_files}")
    print(f"Files to get:         {files_to_get}")
    print(f"Skipping:             {files_to_skip}")
    print(f"Download workers:     {workers}")
    print(f"Batch size:           {write_batch_size:,}")
    print(f"Ngram size:           {ngram_size}")
    print(f"Ngram type:           {ngram_type}")
    print(f"Overwrite DB:         {overwrite_db}")
    print(f"DB Profile:           {open_type}")
    print()
    print(format_banner("Download Progress"))


def print_final_summary(
    start_time: datetime,
    end_time: datetime,
    success: List[str],
    failure: List[str],
    written: int,
    batches: int,
    uncompressed_bytes: int,
) -> None:
    """
    Print final pipeline statistics.

    Args:
        start_time: Pipeline start timestamp
        end_time: Pipeline end timestamp
        success: List of successful file messages
        failure: List of failed file messages
        written: Total entries written
        batches: Number of write batches flushed
        uncompressed_bytes: Total bytes of uncompressed data
    """
    total_runtime = end_time - start_time
    ok = len(success)
    bad = len(failure)

    # Compute per-file statistics
    time_per_file = (total_runtime / ok) if ok else timedelta(0)
    fph = (3600 / time_per_file.total_seconds()) if ok and time_per_file.total_seconds() > 0 else 0.0

    # Compute throughput
    mb_per_sec = 0.0
    if total_runtime.total_seconds() > 0:
        mb_per_sec = (uncompressed_bytes / (1024 * 1024)) / total_runtime.total_seconds()

    print("\nProcessing complete!")
    print()
    print(format_banner("Final Summary"))
    print(f"Fully processed files:       {ok}")
    print(f"Failed files:                {bad}")
    print(f"Total entries written:       {written:,}")
    print(f"Write batches flushed:       {batches}")
    print(f"Uncompressed data processed: {format_bytes(uncompressed_bytes)}")
    print(f"Processing throughput:       {mb_per_sec:.2f} MB/sec")
    print()
    print(f"End Time: {end_time}")
    print(f"Total Runtime: {total_runtime}")
    print(f"Time per file: {time_per_file}")
    print(f"Files per hour: {fph:.1f}")
