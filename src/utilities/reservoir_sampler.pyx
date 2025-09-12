import cython
import signal
import sys
import time as py_time
from libc.stdlib cimport rand, srand
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
        key_type="check",
        return_keys: bool = False,
):
    """
    Alternative version using context manager for automatic cleanup.
    More Pythonic but may have slight overhead.
    """
    cdef list reservoir = []
    cdef int idx
    cdef long long total_processed = 0
    cdef long long skipped_metadata = 0
    cdef long long next_progress = progress_interval

    def signal_handler(signum, frame):
        print(f"\nInterrupt received (signal {signum})")
        print("Cleanup complete. Exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Seed the random number generator
    srand(int(time(NULL)))
    start_time = py_time.time()

    print("=" * 60)
    print(f"RESERVOIR SAMPLING CONFIGURATION")
    print("-" * 60)
    print(f"Target sample size:     {sample_size:,} items")
    print(f"Key handling strategy:  {key_type}")
    print(f"Progress interval:      {progress_interval:,} items")
    if max_items is not None:
        print(f"Database limit:         {max_items:,} entries")
    else:
        print(f"Database limit:         No limit (full traversal)")
    print("=" * 60)

    try:
        with rs.open(db_path, mode="ro") as db:
            iterator = db.iterator()
            iterator.seek(b"")

            while iterator.valid():
                if max_items is not None and (total_processed + skipped_metadata) >= max_items:
                    print(f"[INFO] Reached traversal limit of {max_items:,} entries")
                    break

                key_bytes = iterator.key()
                value_bytes = iterator.value()

                # Handle key type conversion
                if key_type == "string":
                    if isinstance(key_bytes, bytes):
                        key_str = key_bytes.decode('utf-8')
                    else:
                        key_str = str(key_bytes)
                elif key_type == "bytes":
                    key_str = key_bytes.decode('utf-8')
                else:
                    if isinstance(key_bytes, bytes):
                        key_str = key_bytes.decode('utf-8')
                    else:
                        key_str = str(key_bytes)

                # Skip metadata keys
                if key_str.startswith('__'):
                    skipped_metadata += 1
                    iterator.next()
                    continue

                total_processed += 1

                if total_processed >= next_progress:
                    print(f"[PROGRESS] Processed {total_processed:,} items", flush=True)
                    next_progress += progress_interval

                # Reservoir sampling algorithm
                if len(reservoir) < sample_size:
                    if return_keys:
                        reservoir.append((key_str, value_bytes))
                    else:
                        reservoir.append(value_bytes)
                else:
                    idx = rand() % total_processed
                    if idx < sample_size:
                        if return_keys:
                            reservoir[idx] = (key_str, value_bytes)
                        else:
                            reservoir[idx] = value_bytes

                iterator.next()

            del iterator

            # Results reporting
            end_time = py_time.time()
            elapsed_time = end_time - start_time

            print("=" * 60)
            print("RESERVOIR SAMPLING RESULTS")
            print("-" * 60)
            print(f"Items processed:        {total_processed:,}")
            print(f"Metadata entries:       {skipped_metadata:,}")
            print(f"Final sample size:      {len(reservoir):,}")
            print(f"Execution time:         {elapsed_time:.4f} seconds")
            print("-" * 60)

            if elapsed_time > 0:
                items_per_second = total_processed / elapsed_time
                microseconds_per_item = (elapsed_time * 1_000_000) / total_processed if total_processed > 0 else 0
                print("PERFORMANCE METRICS")
                print("-" * 60)
                print(f"Processing rate:        {items_per_second:,.0f} items/second")
                print(f"Time per item:          {microseconds_per_item:.2f} microseconds")
            print("=" * 60)

            return reservoir

    except KeyboardInterrupt:
        print(f"\nInterrupted! Processed {total_processed:,} items so far.")
        return reservoir
    except Exception as e:
        print(f"Error: {e}")
        raise