import cython
import signal
import sys
import time as py_time
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time
import rocks_shim as rs

# Cython-specific optimizations
@cython.boundscheck(False)
@cython.wraparound(False)
def reservoir_sampling(
        db_path,
        sample_size,
        progress_interval=10000000,
        max_items=None,
        return_keys: bool = False,
):
    """
    Perform reservoir sampling on a RocksDB database.

    Args:
        db_path: Path to the RocksDB database
        sample_size: Number of samples to collect
        progress_interval: Print progress every N items
        max_items: Optional limit on items to process
        return_keys: If True, return (key_bytes, value_bytes) tuples; else just values

    Returns:
        List of samples (either values or (key, value) tuples)
    """
    cdef list reservoir = []
    cdef int idx
    cdef long long total_processed = 0
    cdef long long skipped_metadata = 0
    cdef long long next_progress
    cdef int reservoir_len
    cdef int rand_val
    cdef double rand_scale
    cdef long long scaled_idx
    cdef long long estimated_total = 0
    cdef double percent_complete
    cdef int last_percent_reported = 0

    def signal_handler(signum, frame):
        print(f"\nInterrupt received (signal {signum})")
        print("Cleanup complete. Exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Seed the random number generator
    srand(int(time(NULL)))

    # Pre-compute random scaling factor
    rand_scale = 1.0 / <double> RAND_MAX

    # Get database size for percentage-based progress
    with rs.open(db_path, mode="ro") as db:
        total_records_str = db.get_property("rocksdb.estimate-num-keys")
        if total_records_str:
            estimated_total = int(total_records_str)
        else:
            estimated_total = 0  # Will fall back to item-count progress

    # Adaptive progress interval - better for large datasets
    cdef long long adaptive_progress = max(1000000, sample_size * 100)
    next_progress = adaptive_progress

    start_time = py_time.time()

    print("  " + "━" * 60)
    print("  RESERVOIR SAMPLING CONFIGURATION")
    print("  " + "─" * 60)
    print(f"  Target sample size:     {sample_size:,} items")
    if estimated_total > 0:
        print(f"  Database size:          {estimated_total:,} items")
        print(f"  Progress updates:       Every 5% (20 total)")
    else:
        print(f"  Progress interval:      {adaptive_progress:,} items")
    if max_items is not None:
        print(f"  Database limit:         {max_items:,} entries")
    else:
        print(f"  Database limit:         No limit (full traversal)")
    print("  " + "─" * 60)

    try:
        with rs.open(db_path, mode="ro") as db:
            iterator = db.iterator()
            iterator.seek(b"")

            while iterator.valid():
                if max_items is not None and (total_processed + skipped_metadata) >= max_items:
                    print(f"  Reached traversal limit of {max_items:,} entries")
                    break

                key_bytes = iterator.key()

                # Optimized metadata check - direct byte comparison
                if len(key_bytes) >= 2 and key_bytes[0] == 95 and key_bytes[1] == 95:  # 95 = ord('_')
                    skipped_metadata += 1
                    iterator.next()
                    continue

                total_processed += 1

                if total_processed >= next_progress:
                    if estimated_total > 0:
                        # Percentage-based progress
                        percent_complete = (total_processed * 100.0) / estimated_total
                        current_percent = <int> percent_complete

                        # Only report every 5% to avoid spam
                        if current_percent >= last_percent_reported + 5:
                            print(f"  Progress: {total_processed:,} items ({percent_complete:.1f}%)", flush=True)
                            last_percent_reported = current_percent
                            next_progress = total_processed + adaptive_progress
                    else:
                        # Fall back to item-count progress if size unknown
                        print(f"  Processed {total_processed:,} items", flush=True)
                        next_progress += adaptive_progress

                # Cache reservoir length to avoid repeated function calls
                reservoir_len = len(reservoir)

                # Reservoir sampling algorithm
                if reservoir_len < sample_size:
                    # Still filling reservoir - get value only when needed
                    if return_keys:
                        value_bytes = iterator.value()
                        reservoir.append((key_bytes, value_bytes))
                    else:
                        value_bytes = iterator.value()
                        reservoir.append(value_bytes)
                else:
                    # Optimized random selection - avoid modulo operation
                    rand_val = rand()
                    scaled_idx = <long long> (rand_val * rand_scale * total_processed)
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
                microseconds_per_item = (elapsed_time * 1_000_000) / total_processed if total_processed > 0 else 0
                print("  PERFORMANCE METRICS")
                print("  " + "─" * 60)
                print(f"  Processing rate:        {items_per_second:,.0f} items/second")
                print(f"  Time per item:          {microseconds_per_item:.2f} microseconds")
            print("  " + "━" * 60)

            return reservoir

    except KeyboardInterrupt:
        print(f"\nInterrupted! Processed {total_processed:,} items so far.")
        return reservoir
    except Exception as e:
        print(f"Error: {e}")
        raise