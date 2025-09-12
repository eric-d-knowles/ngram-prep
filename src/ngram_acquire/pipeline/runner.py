# ngram_acquire/pipeline/runner.py
from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import PurePosixPath
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type

from tqdm import tqdm
import rocks_shim as rs  # <-- rocks-shim only

from ngram_acquire.pipeline.worker import process_and_ingest_file
from ngram_acquire.db.metadata import mark_file_as_processed
from ngram_acquire.db.write import write_batch_to_db  # <-- rocks-shim strict batching

logger = logging.getLogger(__name__)

DEFAULT_WRITE_BATCH_SIZE = 50_000           # entries (keys)
DEFAULT_WRITE_BATCH_BYTES = 64 * (1 << 20)  # ~64 MiB (keys+values)


def process_files(
        urls: Iterable[str],
        executor_class: Type,  # ThreadPoolExecutor | ProcessPoolExecutor
        workers: int,
        db: rs.DB,
        *,
        filter_pred: Optional[Callable[[str], bool]] = None,
        write_batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
        write_batch_bytes: int = DEFAULT_WRITE_BATCH_BYTES,
) -> Tuple[List[str], List[str], int, int, int]:
    """
    Process files concurrently and ingest results into RocksDB via rocks-shim.

    Returns
    -------
    (success_msgs, failure_msgs, total_entries_written, write_batches, total_uncompressed_bytes)
    """

    # Get the log file path from the current logger configuration
    log_file_path = None
    for handler in logging.getLogger().handlers:
        if isinstance(handler, (logging.FileHandler, logging.handlers.RotatingFileHandler)):
            log_file_path = handler.baseFilename
            break

    logger.info(f"Log file path for workers: {log_file_path}")

    success_msgs: List[str] = []
    failure_msgs: List[str] = []
    total_entries_written = 0
    write_batches = 0
    total_uncompressed_bytes = 0  # Track total uncompressed bytes processed

    pending_data: Dict[str, bytes] = {}
    pending_files: List[str] = []
    pending_bytes: int = 0

    def _approx_kv_bytes(d: Dict[str, bytes]) -> int:
        return sum(len(k.encode("utf-8")) + len(v) for k, v in d.items())

    def flush() -> None:
        nonlocal total_entries_written, write_batches, pending_bytes
        if not pending_data:
            return
        try:
            entries_written = write_batch_to_db(db, pending_data)
            total_entries_written += entries_written
            write_batches += 1
            for fname in pending_files:
                mark_file_as_processed(db, fname)
            logger.info("Flushed batch: %d entries, %d files", entries_written, len(pending_files))
        except Exception as exc:
            msg = f"DB_WRITE_ERROR: {exc}"
            failure_msgs.extend(f"{msg} ({fn})" for fn in pending_files)
            logger.error("%s; aborting pipeline to avoid data loss", msg)
            raise
        else:
            pending_data.clear()
            pending_files.clear()
            pending_bytes = 0

    total = len(urls) if hasattr(urls, "__len__") else None

    with tqdm(total=total, desc="Processing Files", unit="files", colour="blue") as pbar:
        kwargs = {"max_workers": workers}
        if issubclass(executor_class, ProcessPoolExecutor):
            kwargs["mp_context"] = mp.get_context("spawn")
            logger.info(
                "Using multiprocessing start method: %s",
                kwargs["mp_context"].get_start_method()
            )

        with executor_class(**kwargs) as executor:
            it = iter(urls)
            futures: Dict[object, str] = {}
            max_in_flight = max(1, workers * 2)
            idx = 0

            def submit_next(n: int = 1) -> None:
                nonlocal idx
                for _ in range(n):
                    try:
                        url = next(it)
                    except StopIteration:
                        return
                    idx += 1
                    # Pass the log file path to the worker
                    fut = executor.submit(
                        process_and_ingest_file,
                        url,
                        idx,
                        filter_pred,
                        log_file_path
                    )
                    futures[fut] = url

            submit_next(max_in_flight)

            while futures:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    url = futures.pop(fut)
                    filename = PurePosixPath(url).name
                    try:
                        # Unpack the new 3-tuple return format
                        result_msg, parsed_data, uncompressed_bytes = fut.result()
                        if result_msg.startswith("SUCCESS"):
                            success_msgs.append(result_msg)
                            # Accumulate uncompressed bytes from successful workers
                            total_uncompressed_bytes += uncompressed_bytes
                            pending_data.update(parsed_data)
                            pending_files.append(filename)
                            pending_bytes += _approx_kv_bytes(parsed_data)
                            if (
                                    len(pending_data) >= write_batch_size
                                    or pending_bytes >= write_batch_bytes
                            ):
                                flush()
                            logger.info("Processed: %s", filename)
                        else:
                            failure_msgs.append(result_msg)
                            # Note: uncompressed_bytes will be 0 for failed files
                    except Exception as exc:
                        msg = f"ERROR: {filename} - {exc}"
                        failure_msgs.append(msg)
                        logger.error(msg)
                    finally:
                        pbar.update(1)

                submit_next(len(done))

    flush()
    return success_msgs, failure_msgs, total_entries_written, write_batches, total_uncompressed_bytes