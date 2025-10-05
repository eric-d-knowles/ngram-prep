"""Optimized reservoir sampling for RocksDB databases."""

import cython
import signal
import sys
import time as py_time
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time
import rocks_shim as rs

__all__ = ["reservoir_sampling"]

@cython.boundscheck(False)
@cython.wraparound(False)
def reservoir_sampling(
        db_path,
        sample_size,
        total_records=None,
        progress_interval=5,
        return_keys: bool = False,
        max_items=None,
):
    """
    Perform reservoir sampling on a RocksDB database.

    Uses optimized Cython implementation with low-level C operations
    for maximum performance on large databases.

    Args:
        db_path: Path to the RocksDB database
        sample_size: Number of samples to collect
        total_records: Estimated total records (for progress reporting)
        progress_interval: Report progress every N percentage points
        return_keys: If True, return (key, value) tuples; else just values
        max_items: Optional limit on items to process

    Returns:
        List of samples (either values or (key, value) tuples)
    """
    cdef list reservoir = []
    cdef int idx
    cdef long long total_processed = 0
    cdef long long skipped_metadata = 0
    cdef int reservoir_len
    cdef int rand_val
    cdef double rand_scale
    cdef long long scaled_idx
    cdef long long estimated_total = (
        total_records if total_records else 0
    )
    cdef double percent_complete
    cdef double next_percent_milestone = progress_interval

    def signal_handler(signum, frame):
        """Handle interrupt signals gracefully."""
        print(f"\nInterrupt received (signal {signum})")
        print("Cleanup complete. Exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Seed the random number generator
    srand(int(time(NULL)))

    # Pre-compute random scaling factor
    rand_scale = 1.0 / <double> RAND_MAX

    start_time = py_time.time()

    # Print configuration
    print("  " + "━" * 60)
    print("  RESERVOIR SAMPLING CONFIGURATION")
    print("  " + "─" * 60)
    print(f"  Target sample size:     {sample_size:,} items")
    if estimated_total > 0:
        print(f"  Database size:          {estimated_total:,} items")
        print(
            f"  Progress reporting:     Every {progress_interval}% complete"
        )
    else:
        print("  Database size:          Unknown (will report every 1M items)")
    if max_items is not None:
        print(f"  Database limit:         {max_items:,} entries")
    else:
        print("  Database limit:         No limit (full traversal)")
    print("  " + "─" * 60)

    try:
        with rs.open(db_path, mode="r") as db:
            iterator = db.iterator()
            iterator.seek(b"")

            while iterator.valid():
                # Check traversal limit
                if (
                        max_items is not None
                        and (total_processed + skipped_metadata) >= max_items
                ):
                    print(f"  Reached traversal limit of {max_items:,} entries")
                    break

                key_bytes = iterator.key()

                # Optimized metadata check - direct byte comparison
                # Skip keys starting with "__" (95 = ord('_'))
                if (
                        len(key_bytes) >= 2
                        and key_bytes[0] == 95
                        and key_bytes[1] == 95
                ):
                    skipped_metadata += 1
                    iterator.next()
                    continue

                total_processed += 1

                # Progress reporting logic
                if estimated_total > 0:
                    percent_complete = (total_processed * 100.0) / estimated_total
                    if percent_complete >= next_percent_milestone:
                        print(
                            f"  Progress: {percent_complete:.1f}% ({total_processed:,} items)",
                            flush=True
                        )
                        next_percent_milestone += progress_interval
                else:
                    # Fallback for unknown size - report every million
                    if total_processed % 1_000_000 == 0:
                        print(f"  Processed {total_processed:,} items", flush=True)

                # Cache reservoir length to avoid repeated calls
                reservoir_len = len(reservoir)

                # Reservoir sampling algorithm
                if reservoir_len < sample_size:
                    # Still filling reservoir
                    if return_keys:
                        value_bytes = iterator.value()
                        reservoir.append((key_bytes, value_bytes))
                    else:
                        value_bytes = iterator.value()
                        reservoir.append(value_bytes)
                else:
                    # Optimized random selection
                    rand_val = rand()
                    scaled_idx = <long long> (
                            rand_val * rand_scale * total_processed
                    )
                    if scaled_idx < sample_size:
                        idx = <int> scaled_idx
                        if return_keys:
                            value_bytes = iterator.value()
                            reservoir[idx] = (key_bytes, value_bytes)
                        else:
                            value_bytes = iterator.value()
                            reservoir[idx] = value_bytes

                iterator.next()

            del iterator

            # Results reporting
            end_time = py_time.time()
            elapsed_time = end_time - start_time

            print("  " + "─" * 60)
            print("  RESERVOIR SAMPLING RESULTS")
            print("  " + "─" * 60)
            print(f"  Items processed:        {total_processed:,}")
            print(f"  Metadata entries:       {skipped_metadata:,}")
            print(f"  Final sample size:      {len(reservoir):,}")
            print(f"  Execution time:         {elapsed_time:.4f} seconds")
            print("  " + "─" * 60)

            if elapsed_time > 0:
                items_per_second = total_processed / elapsed_time
                microseconds_per_item = (
                    (elapsed_time * 1_000_000) / total_processed
                    if total_processed > 0
                    else 0
                )
                print("  PERFORMANCE METRICS")
                print("  " + "─" * 60)
                print(
                    f"  Processing rate:        "
                    f"{items_per_second:,.0f} items/second"
                )
                print(
                    f"  Time per item:          "
                    f"{microseconds_per_item:.2f} microseconds"
                )
            print("  " + "━" * 60)

            return reservoir

    except KeyboardInterrupt:
        print(f"\nInterrupted! Processed {total_processed:,} items so far.")
        return reservoir
    except Exception as e:
        print(f"Error during sampling: {e}")
        raise