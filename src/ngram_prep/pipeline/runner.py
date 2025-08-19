from __future__ import annotations

import logging
from concurrent.futures import as_completed
from pathlib import PurePosixPath
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type

from tqdm import tqdm

from rocksdict import Rdict  # type: ignore

from ngram_prep.pipeline.worker import process_and_ingest_file
from ngram_prep.db.metadata import mark_file_as_processed
# write_batch_to_db: your existing helper that writes {key: bytes} to RocksDB
from ngram_prep.db.write import write_batch_to_db  # adjust import as needed

logger = logging.getLogger(__name__)

DEFAULT_WRITE_BATCH_SIZE = 50_000  # or import from your constants


def process_files(
    urls: Iterable[str],
    executor_class: Type,  # ThreadPoolExecutor | ProcessPoolExecutor
    workers: int,
    db: Rdict,
    *,
    filter_pred: Optional[Callable[[str], bool]] = None,
    write_batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
) -> Tuple[List[str], List[str], int, int]:
    """
    Process files concurrently; return (success_msgs, failure_msgs, entries, batches).

    Notes
    -----
    - Files are marked 'processed' only after the batch write that includes them
      succeeds, to make resume semantics correct.
    - write_batch_size counts *entries* (unique ngram keys), not bytes.
    """
    success_msgs: List[str] = []
    failure_msgs: List[str] = []
    total_entries_written = 0
    write_batches = 0

    pending_data: Dict[str, bytes] = {}
    pending_files: List[str] = []  # filenames whose data has been staged

    def flush() -> None:
        nonlocal total_entries_written, write_batches
        if not pending_data:
            return
        try:
            entries_written = write_batch_to_db(db, pending_data)
            total_entries_written += entries_written
            write_batches += 1
            # Mark *after* a successful write
            for fname in pending_files:
                mark_file_as_processed(db, fname)
            logger.info(
                "Flushed batch: %d entries, %d files",
                entries_written,
                len(pending_files),
            )
        except Exception as exc:  # pragma: no cover (depends on your write path)
            # Conservatively do not mark files; record an error message per file
            msg = f"DB_WRITE_ERROR: {exc}"
            failure_msgs.extend(f"{msg} ({fn})" for fn in pending_files)
            logger.error("%s; batch of %d files dropped", msg, len(pending_files))
        finally:
            pending_data.clear()
            pending_files.clear()

    urls = list(urls)
    with tqdm(total=len(urls), desc="Processing Files", unit="files",
              colour="blue") as pbar:
        with executor_class(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_and_ingest_file, url, idx, filter_pred
                ): url
                for idx, url in enumerate(urls, start=1)
            }

            for fut in as_completed(futures):
                url = futures[fut]
                filename = PurePosixPath(url).name
                try:
                    result_msg, parsed_data = fut.result()
                    if result_msg.startswith("SUCCESS"):
                        success_msgs.append(result_msg)
                        # Stage data; mark will occur on flush
                        pending_data.update(parsed_data)
                        pending_files.append(filename)
                        if len(pending_data) >= write_batch_size:
                            flush()
                        logger.info("Processed: %s", filename)
                    else:
                        failure_msgs.append(result_msg)
                except Exception as exc:
                    msg = f"ERROR: {filename} - {exc}"
                    failure_msgs.append(msg)
                    logger.error(msg)
                finally:
                    pbar.update(1)

    # Final flush after all workers complete
    flush()

    return success_msgs, failure_msgs, total_entries_written, write_batches
