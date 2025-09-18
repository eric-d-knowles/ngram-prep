# ngram_acquire/pipeline/orchestrate.py
from __future__ import annotations

import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import PurePosixPath
from typing import Optional, Tuple, Type

try:
    import setproctitle as _setproctitle  # optional nicety
except Exception:  # pragma: no cover
    _setproctitle = None

from ngram_acquire.io.locations import set_location_info
from ngram_acquire.io.fetch import fetch_file_urls
from ngram_acquire.utils.filters import make_ngram_type_predicate
from ngram_acquire.pipeline.report import print_run_summary
from ngram_acquire.pipeline.runner import process_files
from ngram_acquire.db.metadata import is_file_processed
from ngram_acquire.db.write import DEFAULT_WRITE_BATCH_SIZE
from ngram_acquire.utils.cleanup import safe_db_cleanup
from common_db.api import open_db

logger = logging.getLogger(__name__)


def _format_bytes(byte_count: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if byte_count < 1024.0:
            if unit == 'bytes':
                return f"{byte_count:,} {unit}"
            else:
                return f"{byte_count:.2f} {unit}"
        byte_count /= 1024.0
    return f"{byte_count:.2f} PB"


def download_and_ingest_to_rocksdb(
        ngram_size: int,
        repo_release_id: str,
        repo_corpus_id: str,
        db_path: str,
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
    """Discover files, download/parse, and ingest into RocksDB (via rocks-shim)."""
    logger.info("Starting N-gram processing pipeline")

    if _setproctitle is not None:  # pragma: no cover
        try:
            _setproctitle.setproctitle("PROC_MAIN")
        except Exception:
            pass

    start_time = datetime.now()

    # Worker count
    if workers is None:
        cpu = os.cpu_count() or 4
        workers = min(40, cpu * 2)

    # Overwrite existing DB dir
    if overwrite and os.path.exists(db_path):
        logger.info("Removing existing database for fresh start…")
        if not safe_db_cleanup(db_path):
            raise RuntimeError(
                f"Failed to remove existing database at {db_path}. "
                "Close open handles or remove it manually."
            )
        logger.info("Successfully removed existing database")

    # Ensure parent dir exists
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # 1) Discover files
    page_url, file_rx = set_location_info(ngram_size, repo_release_id, repo_corpus_id)
    file_urls_available = fetch_file_urls(page_url, file_rx)
    if not file_urls_available:
        raise RuntimeError("No n-gram files found in the repository")

    # 2) Select subset
    if file_range is None:
        file_range = (0, len(file_urls_available) - 1)

    start_idx, end_idx = file_range
    if start_idx < 0 or end_idx >= len(file_urls_available) or start_idx > end_idx:
        raise ValueError(
            f"Invalid file range {file_range}. Available: 0..{len(file_urls_available) - 1}"
        )
    file_urls_to_use = file_urls_available[start_idx : end_idx + 1]

    # 3) Open DB with rocks-shim
    ot = (open_type).lower()
    if ot not in {"read", "write", "read:packed24", "write:packed24"}:
        raise ValueError("open_type must be one of 'read', 'write', 'read:packed24', 'write:packed24'")
    profile = ot

    if ot == "read":
        logger.info("Using read-optimized RocksDB options")
    elif ot == "write":
        logger.info("Using write-optimized RocksDB options")
    elif ot == "read:packed24":
        logger.info("Using read-optimized RocksDB options (packed24)")
    elif ot == "write:packed24":
        logger.info("Using write-optimized RocksDB options (packed24)")

    # Add this debug code right before "with open_db(db_path, profile=profile) as db:"
    #print(f"DEBUG orchestrate.py: open_type='{open_type}', ot='{ot}', profile='{profile}'")

    with open_db(db_path, profile=profile) as db:
        # Add this debug check right after opening
        #if hasattr(db, 'get_property'):
        #    try:
        #        auto_compact = db.get_property("rocksdb.disable-auto-compactions")
        #        print(f"DEBUG: After opening, disable_auto_compactions = {auto_compact}")
        #    except Exception as e:
        #        print(f"DEBUG: Could not read property: {e}")
        # Resume filter
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
                logger.info("Resume mode: skipping %s processed files", files_to_skip)
            if not file_urls_to_use:
                print("🎉 All files in the specified range are already processed!")
                return  # context closes DB

        # Optional shuffle
        if random_seed is not None:
            random.seed(random_seed)
            random.shuffle(file_urls_to_use)
            logger.info("Randomized file order with seed %s", random_seed)

        # Executor & predicate
        executor_class: Type = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        executor_name = "threads" if use_threads else "processes"
        filter_pred = make_ngram_type_predicate(ngram_type)

        # 3.5) Summary
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
            color=True,
        )

        # 4) Process & ingest - now returns uncompressed byte count
        success, failure, written, batches, uncompressed_bytes = process_files(
            urls=file_urls_to_use,
            executor_class=executor_class,
            workers=workers,
            db=db,
            filter_pred=filter_pred,
            write_batch_size=write_batch_size,
        )

        if post_compact:
            logger.info("Bulk ingestion complete. Starting manual compaction...")
            print("\n\033[33mStarting post-ingest compaction...\033[0m")

            compact_start = datetime.now()
            # Perform full manual compaction
            db.compact_all()
            logger.info("Manual compaction completed successfully")

            compact_end = datetime.now()
            compact_time = compact_end - compact_start
            print(f"\033[32mCompaction completed in {compact_time}\033[0m")

    # 5) Report stats
    end_time = datetime.now()
    total_runtime = end_time - start_time
    ok = len(success)
    bad = len(failure)
    time_per_file = (total_runtime / ok) if ok else total_runtime
    fph = (3600 / time_per_file.total_seconds()) if ok else 0.0

    print("\033[32m\nProcessing completed!\033[0m")
    print(f"Fully processed files: {ok}")
    if bad:
        print(f"\033[31mFailed files: {bad}\033[0m")
    print(f"Total entries written: {written:,}")
    print(f"Write batches flushed: {batches}")

    # New: Display uncompressed data processed
    print(f"Uncompressed data processed: {_format_bytes(uncompressed_bytes)}")

    # Calculate and display processing throughput
    if total_runtime.total_seconds() > 0:
        mb_per_sec = (uncompressed_bytes / (1024 * 1024)) / total_runtime.total_seconds()
        print(f"Processing throughput: {mb_per_sec:.2f} MB/sec")

    print(f"\033[31m\nEnd Time: {end_time}\033[0m")
    print(f"\033[31mTotal Runtime: {total_runtime}\033[0m")
    print(f"\033[34m\nTime per file: {time_per_file}\033[0m")
    print(f"\033[34mFiles per hour: {fph:.1f}\033[0m")