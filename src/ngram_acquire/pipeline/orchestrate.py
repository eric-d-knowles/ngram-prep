"""Main orchestration for ngram acquisition pipeline."""
from __future__ import annotations

import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import PurePosixPath
from typing import Optional, Tuple, Type

from ngram_acquire.io.locations import build_location_info
from ngram_acquire.io.fetch import fetch_file_urls
from ngram_acquire.utils.filters import make_ngram_type_predicate
from ngram_acquire.pipeline.report import print_run_summary
from ngram_acquire.pipeline.runner import process_files
from ngram_acquire.db.metadata import is_file_processed
from ngram_acquire.db.write import DEFAULT_WRITE_BATCH_SIZE
from ngram_acquire.utils.cleanup import safe_db_cleanup
from ngram_acquire.db.build_path import build_db_path
from common_db.api import open_db

logger = logging.getLogger(__name__)

__all__ = ["download_and_ingest_to_rocksdb"]

try:
    import setproctitle as _setproctitle
except ImportError:
    _setproctitle = None


def _format_bytes(byte_count: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["bytes", "KB", "MB", "GB", "TB"]:
        if byte_count < 1024.0:
            if unit == "bytes":
                return f"{byte_count:,} {unit}"
            return f"{byte_count:.2f} {unit}"
        byte_count /= 1024.0
    return f"{byte_count:.2f} PB"


def download_and_ingest_to_rocksdb(
        ngram_size: int,
        repo_release_id: str,
        repo_corpus_id: str,
        db_path_stub: str,
        file_range: Optional[Tuple[int, int]] = None,
        workers: Optional[int] = None,
        use_threads: bool = False,
        ngram_type: str = "all",
        overwrite: bool = True,
        random_seed: Optional[int] = None,
        write_batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
        open_type: str = "read",
        post_compact: bool = False,
) -> None:
    """
    Main pipeline: discover, download, parse, and ingest ngram files into RocksDB.

    Orchestrates the complete ngram acquisition workflow:
    1. Builds database path from stub and parameters
    2. Discovers available files from repository
    3. Opens/creates RocksDB with specified profile
    4. Downloads and processes files concurrently
    5. Writes batched results to database
    6. Optionally performs post-ingestion compaction

    Args:
        ngram_size: N-gram size (1-5)
        repo_release_id: Release date in YYYYMMDD format (e.g., "20200217")
        repo_corpus_id: Corpus identifier (e.g., "eng", "eng-us")
        db_path_stub: Base directory for database (will be expanded)
        file_range: Optional (start_idx, end_idx) to process subset of files
        workers: Number of concurrent workers (default: min(40, cpu_count * 2))
        use_threads: If True, use threads; otherwise use processes
        ngram_type: Filter type ("all", "tagged", etc.)
        overwrite: If True, remove existing database before starting
        random_seed: Optional seed for randomizing file processing order
        write_batch_size: Number of entries per batch write
        open_type: RocksDB profile ("read", "write", "read:packed24", "write:packed24")
        post_compact: If True, run manual compaction after ingestion
    """
    logger.info("Starting N-gram processing pipeline")

    # Set process title if available
    if _setproctitle is not None:
        try:
            _setproctitle.setproctitle("PROC_MAIN")
        except Exception:
            pass

    start_time = datetime.now()

    # Build full database path from stub
    db_path = build_db_path(
        db_path_stub, ngram_size, repo_release_id, repo_corpus_id
    )
    logger.info("Database path: %s", db_path)

    # Determine worker count
    if workers is None:
        cpu = os.cpu_count() or 4
        workers = min(40, cpu * 2)

    # Handle existing database
    if overwrite and os.path.exists(db_path):
        logger.info("Removing existing database for fresh start")
        if not safe_db_cleanup(db_path):
            raise RuntimeError(
                f"Failed to remove existing database at {db_path}. "
                "Close open handles or remove it manually."
            )
        logger.info("Successfully removed existing database")

    # Ensure parent directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # Discover available files
    page_url, file_rx = build_location_info(ngram_size, repo_release_id, repo_corpus_id)
    file_urls_available = fetch_file_urls(page_url, file_rx)
    if not file_urls_available:
        raise RuntimeError("No n-gram files found in the repository")

    # Select file subset
    if file_range is None:
        file_range = (0, len(file_urls_available) - 1)

    start_idx, end_idx = file_range
    if start_idx < 0 or end_idx >= len(file_urls_available) or start_idx > end_idx:
        raise ValueError(
            f"Invalid file range {file_range}. Available: 0..{len(file_urls_available) - 1}"
        )
    file_urls_to_use = file_urls_available[start_idx: end_idx + 1]

    # Validate and normalize open_type
    open_type = open_type.lower()
    valid_profiles = {"read", "write", "read:packed24", "write:packed24"}
    if open_type not in valid_profiles:
        raise ValueError(
            f"open_type must be one of {valid_profiles}, got {open_type!r}"
        )

    logger.info("Using RocksDB profile: %s", open_type)

    # Open database with specified profile
    with open_db(db_path, profile=open_type) as db:
        # Resume mode: skip already-processed files
        files_to_skip = 0
        if not overwrite:
            to_keep = []
            for url in file_urls_to_use:
                name = PurePosixPath(url).name
                if not is_file_processed(db, name):
                    to_keep.append(url)
            files_to_skip = len(file_urls_to_use) - len(to_keep)
            file_urls_to_use = to_keep

            if files_to_skip:
                logger.info("Resume mode: skipping %d processed files", files_to_skip)

            if not file_urls_to_use:
                print("All files in the specified range are already processed!")
                return

        # Optional randomization
        if random_seed is not None:
            random.seed(random_seed)
            random.shuffle(file_urls_to_use)
            logger.info("Randomized file order with seed %d", random_seed)

        # Configure executor
        executor_class: Type = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        executor_name = "threads" if use_threads else "processes"
        filter_pred = make_ngram_type_predicate(ngram_type)

        # Print run summary
        print_run_summary(
            ngram_repo_url=page_url,
            db_path=db_path,
            file_range=(start_idx, end_idx),
            file_urls_available=file_urls_available,
            file_urls_to_use=file_urls_to_use,
            ngram_size=ngram_size,
            workers=workers,
            executor_name=executor_name,
            start_time=start_time,
            ngram_type=ngram_type,
            overwrite=overwrite,
            files_to_skip=files_to_skip,
            write_batch_size=write_batch_size,
            profile=open_type,
            post_compact=post_compact,
        )

        # Process and ingest files
        success, failure, written, batches, uncompressed_bytes = process_files(
            urls=file_urls_to_use,
            executor_class=executor_class,
            workers=workers,
            db=db,
            filter_pred=filter_pred,
            write_batch_size=write_batch_size,
        )

        # Optional post-ingestion compaction
        if post_compact:
            logger.info("Bulk ingestion complete. Starting manual compaction")
            print("Compacting... ", end="", flush=True)

            compact_start = datetime.now()
            db.compact_all()

            compact_end = datetime.now()
            compact_time = compact_end - compact_start
            print(f"completed in {compact_time}")
            logger.info("Manual compaction completed in %s", compact_time)

    # Report final statistics
    end_time = datetime.now()
    total_runtime = end_time - start_time
    ok = len(success)
    bad = len(failure)
    time_per_file = (total_runtime / ok) if ok else total_runtime
    fph = (3600 / time_per_file.total_seconds()) if ok else 0.0
    mb_per_sec = (uncompressed_bytes / (1024 * 1024)) / total_runtime.total_seconds()

    lines = [
        "\nProcessing complete!",
        "\nProcessing Summary",
        "═" * 100,
        f"Fully processed files:       {ok}",
        f"Failed files:                {bad}",
        f"Total entries written:       {written:,}",
        f"Write batches flushed:       {batches}",
        f"Uncompressed data processed: {_format_bytes(uncompressed_bytes)}",
        f"Processing throughput:       {mb_per_sec:.2f} MB/sec",
        f"\nEnd Time: {end_time}",
        f"Total Runtime: {total_runtime}",
        f"Time per file: {time_per_file}",
        f"Files per hour: {fph:.1f}",
    ]

    print("\n".join(lines))