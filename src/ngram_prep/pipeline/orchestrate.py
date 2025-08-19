# ngram_prep/pipeline/orchestrate.py (or wherever you keep the orchestration)
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

from ngram_prep.io.locations import set_location_info
from ngram_prep.io.fetch import fetch_file_urls
from ngram_prep.utils.filters import make_ngram_type_predicate
from ngram_prep.pipeline.report import print_run_summary
from ngram_prep.pipeline.runner import process_files
from ngram_prep.db.setup import setup_rocksdb
from ngram_prep.db.metadata import is_file_processed
from ngram_prep.db.write import DEFAULT_WRITE_BATCH_SIZE
from ngram_prep.utils.cleanup import safe_db_cleanup

# If you have a log monitor, import it; otherwise this stays a no-op.
try:
    from ngram_prep.pipeline.log_monitor import start_rocksdb_log_monitor
except Exception:  # pragma: no cover
    start_rocksdb_log_monitor = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


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
) -> None:
    """
    Discover Google Books n-gram files, download, parse, and ingest into RocksDB.

    Process
    -------
    1. Build repo URL and enumerate files
    2. Select subset (file_range) and optionally randomize
    3. Create/prepare RocksDB (optionally overwrite)
    4. Run concurrent workers; batch writes; mark processed after flush
    5. Print a concise run summary and completion stats
    """
    # Cosmetic: label main process in htop if available
    if _setproctitle is not None:  # pragma: no cover
        try:
            _setproctitle.setproctitle("PROC_MAIN")
        except Exception:
            pass

    start_time = datetime.now()

    # Cap worker count for stability. Threads help with I/O; processes for CPU.
    if workers is None:
        cpu = os.cpu_count() or 4
        workers = min(40, cpu * 2)

    # Overwrite mode: remove any existing DB directory safely (NFS-aware).
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

    # Optional log monitor
    if start_rocksdb_log_monitor is not None:  # pragma: no cover
        try:
            # Typical RocksDB log lives in <db_path>/LOG or <db_dir>/LOG
            log_path = os.path.join(db_path, "LOG")
            if not os.path.exists(log_path):
                log_path = os.path.join(db_dir or ".", "LOG")
            start_rocksdb_log_monitor(log_path, poll_interval=10, logger=logger)
        except Exception as exc:
            logger.warning("Could not start RocksDB log monitor: %s", exc)

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
            f"Invalid file range {file_range}. Available: 0.."
            f"{len(file_urls_available) - 1}"
        )

    file_urls_to_use = file_urls_available[start_idx : end_idx + 1]

    # 3) Open DB. In resume mode, pre-filter URLs against processed markers.
    db = setup_rocksdb(db_path)
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
            db.close()
            return

    # Optional randomization for load spreading across shards/blocks.
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(file_urls_to_use)
        logger.info("Randomized file order with seed %s", random_seed)

    # Executor choice
    executor_class: Type = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    executor_name = "threads" if use_threads else "processes"

    # Filter predicate: fast token screening (no regex overhead).
    filter_pred = make_ngram_type_predicate(ngram_type)

    # Print a run summary (human-friendly)
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
    )

    # 4) Run workers and batch writes (mark processed only after flush)
    success, failure, written, batches = process_files(
        urls=file_urls_to_use,
        executor_class=executor_class,
        workers=workers,
        db=db,
        filter_pred=filter_pred,
        write_batch_size=write_batch_size,
    )

    # 5) Close DB and report stats
    db.close()

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
    print(f"\033[31m\nEnd Time: {end_time}\033[0m")
    print(f"\033[31mTotal Runtime: {total_runtime}\033[0m")
    print(f"\033[34m\nTime per file: {time_per_file}\033[0m")
    print(f"\033[34mFiles per hour: {fph:.1f}\033[0m")
