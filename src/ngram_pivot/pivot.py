# ngram_pivot/pivot.py
"""Core pivot transformation logic."""
from pathlib import Path
from datetime import datetime, timedelta
import logging
import time

from tqdm import tqdm

from common_db.api import open_db, scan_all
from ngram_pivot.config import PivotConfig
from ngram_pivot.encoding import (
    encode_year_ngram_key,
    encode_year_stats,
    decode_packed24_records,
)

logger = logging.getLogger(__name__)

LINE_WIDTH = 100


def _truncate_path_to_fit(
        path: Path | str,
        prefix: str,
        total_width: int = LINE_WIDTH,
) -> str:
    """
    Truncate path to fit within total_width including prefix.

    The total line length (prefix + path) will not exceed total_width.
    Longer prefixes automatically get less space for the path.

    Args:
        path: Path to display
        prefix: The label/prefix before the path
        total_width: Total character width for the entire line

    Returns:
        Truncated path that fits within (total_width - len(prefix))

    Examples:
        >>> _truncate_path_to_fit("/long/path", "Short: ", 50)
        '/long/path'  # Fits within 50 - 7 = 43 chars
        >>> _truncate_path_to_fit("/long/path", "Very long prefix: ", 50)
        '...path'  # Truncated to fit within 50 - 18 = 32 chars
    """
    path_str = str(path)
    max_path_length = total_width - len(prefix)

    if len(path_str) <= max_path_length:
        return path_str

    # Need at least 4 chars for "..."
    if max_path_length < 4:
        return "..."

    return "..." + path_str[-(max_path_length - 3):]


def pivot_database(config: PivotConfig) -> None:
    """
    Transform n-gram-indexed database to year-indexed database.

    Reads from source DB where:
        Key: n-gram (bytes)
        Value: Packed24 format (year, occurrences, documents) tuples

    Writes to target DB where:
        Key: year (4-byte big-endian) + n-gram (bytes)
        Value: (occurrences, documents) as 16-byte packed uint64s

    Args:
        config: Pivot configuration
    """
    start_time = datetime.now()

    # Format paths
    source_path_str = _truncate_path_to_fit(config.source_db_path, "Source DB:            ")
    target_path_str = _truncate_path_to_fit(config.target_db_path, "Target DB:            ")

    # Print run summary
    lines = [
        "N-GRAM DATABASE PIVOT",
        "━" * LINE_WIDTH,
        f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}",
        "",
        "Configuration",
        "═" * LINE_WIDTH,
        f"Source DB:            {source_path_str}",
        f"Target DB:            {target_path_str}",
        f"Source profile:       {config.source_profile}",
        f"Target profile:       {config.target_profile}",
        f"Write batch size:     {config.write_batch_size:,}",
        f"Validation enabled:   {config.validate}",
        "",
        "Progress",
        "═" * LINE_WIDTH,
        ]
    print("\n".join(lines), flush=True)

    logger.info("Starting pivot: %s -> %s", config.source_db_path, config.target_db_path)

    # Validate source exists
    if not config.source_db_path.exists():
        raise FileNotFoundError(f"Source database not found: {config.source_db_path}")

    # Create target directory if needed
    config.target_db_path.parent.mkdir(parents=True, exist_ok=True)

    stats = PivotStats()
    process_start = time.time()

    with open_db(config.source_db_path, mode="r", profile=config.source_profile) as source_db:
        with open_db(
                config.target_db_path,
                mode="rw",
                profile=config.target_profile,
                create_if_missing=True,
        ) as target_db:

            # Accumulate writes in a list, then batch them
            write_buffer = []

            # Progress bar for source records
            with tqdm(
                    desc="Pivoting",
                    unit="ngram",
                    unit_scale=True,
                    ncols=LINE_WIDTH,
                    bar_format='{desc} |{bar}| {n_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            ) as pbar:

                for ngram_key, packed_value in scan_all(source_db):
                    stats.source_records += 1
                    pbar.update(1)

                    try:
                        # Decode all year records for this n-gram
                        year_records = decode_packed24_records(packed_value)
                        stats.total_year_records += len(year_records)

                        # Create one target record per year
                        for year, occurrences, documents in year_records:
                            target_key = encode_year_ngram_key(year, ngram_key)
                            target_value = encode_year_stats(occurrences, documents)

                            write_buffer.append((target_key, target_value))

                            # Flush batch when full
                            if len(write_buffer) >= config.write_batch_size:
                                batch_start = time.time()
                                _flush_batch(target_db, write_buffer, stats)
                                batch_time = time.time() - batch_start
                                write_buffer.clear()

                                # Update progress bar description with batch info
                                pbar.set_description(
                                    f"Pivoting [{stats.target_records:,} written, "
                                    f"batch {batch_time:.1f}s]"
                                )

                    except Exception as e:
                        logger.error("Error processing n-gram %r: %s", ngram_key, e)
                        stats.errors += 1
                        if config.validate:
                            raise

                # Flush any remaining writes
                if write_buffer:
                    _flush_batch(target_db, write_buffer, stats)
                    write_buffer.clear()

    # Final summary
    elapsed_total = time.time() - process_start
    end_time = datetime.now()
    records_per_sec = stats.target_records / elapsed_total if elapsed_total > 0 else 0

    lines = [
        "",
        "Pivot Complete",
        "═" * LINE_WIDTH,
        f"Source n-grams:              {stats.source_records:,}",
        f"Year records found:          {stats.total_year_records:,}",
        f"Target records written:      {stats.target_records:,}",
        f"Errors:                      {stats.errors:,}",
        f"Throughput:                  {records_per_sec:,.0f} records/sec",
        f"Total runtime:               {str(timedelta(seconds=int(elapsed_total)))}",
        "",
        f"End Time:                    {end_time:%Y-%m-%d %H:%M:%S}",
        "━" * LINE_WIDTH,
        ]
    print("\n".join(lines), flush=True)

    logger.info("Pivot complete - processed %s n-grams into %s target records",
                f"{stats.source_records:,}", f"{stats.target_records:,}")


def _flush_batch(db, buffer: list, stats: 'PivotStats') -> None:
    """Flush a batch of writes to the database."""
    batch = db.write_batch(disable_wal=True, sync=False)
    for key, value in buffer:
        batch.put(key, value)
        stats.target_records += 1
    # Batch auto-commits when it goes out of scope
    del batch


class PivotStats:
    """Statistics for pivot operation."""

    def __init__(self):
        self.source_records = 0      # Number of n-grams processed
        self.total_year_records = 0  # Total (year, count, docs) tuples seen
        self.target_records = 0      # Number of year-ngram pairs written
        self.errors = 0              # Number of errors encountered