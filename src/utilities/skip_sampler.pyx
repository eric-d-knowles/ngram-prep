# skip_sampler.pyx
import signal
import sys
import time as py_time
from libc.stdlib cimport rand, srand
from libc.time cimport time
import rocks_shim as rs
import cython

def skip_sampling(
        db_path,
        sample_size,
        max_items=None,
        return_keys: bool = False,
):
    """
    Fast skip-sampling for work unit creation.
    """
    def signal_handler(signum, frame):
        print(f"\nInterrupt received (signal {signum})")
        print("Cleanup complete. Exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    srand(int(time(NULL)))
    start_time = py_time.time()

    # Calculate adaptive progress interval - one order of magnitude less than sample size
    adaptive_progress = max(100, sample_size // 10)  # At least every 100 samples

    print("  " + "━" * 60)
    print("  FAST SKIP SAMPLING CONFIGURATION")
    print("  " + "─" * 60)
    print(f"  Target sample size:     {sample_size:,} items")
    print(f"  Progress updates every: {adaptive_progress:,} samples")
    if max_items is not None:
        print(f"  Database limit:         {max_items:,} entries")
    print("  " + "─" * 60)

    try:
        final_samples = fast_skip_sampling(
            db_path, sample_size, return_keys, adaptive_progress, max_items
        )

        end_time = py_time.time()
        elapsed_time = end_time - start_time

        print("  " + "━" * 60)
        print("  FAST SAMPLING RESULTS")
        print("  " + "─" * 60)
        print(f"  Final sample size:      {len(final_samples):,}")
        print(f"  Total execution time:   {elapsed_time:.1f}s")
        if elapsed_time > 0:
            print(f"  Sampling rate:          {len(final_samples) / elapsed_time:,.0f} samples/second")
        print("  " + "━" * 60)

        return final_samples

    except KeyboardInterrupt:
        print(f"\nInterrupted!")
        return []
    except Exception as e:
        print(f"Error: {e}")
        raise

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_skip_sampling(
        str db_path,
        long long sample_size,
        bint return_keys,
        long long progress_interval,
        max_items
):
    """Super simple skip sampling - examine every Nth item."""

    # Get database size
    with rs.open(db_path, mode="ro") as db:
        total_records_str = db.get_property("rocksdb.estimate-num-keys")
        total_items = int(total_records_str) if total_records_str else 2_500_000_000

    # Calculate skip size
    cdef long long skip_size = max(1, total_items // (sample_size * 2))  # 2x oversampling
    cdef list samples = []
    cdef long long processed = 0
    cdef long long examined = 0
    cdef long long next_progress = progress_interval

    print(f"  Database size: ~{total_items:,} items")
    print(f"  Skip interval: {skip_size:,} (examine every {skip_size:,}th item)")

    with rs.open(db_path, mode="ro") as db:
        iterator = db.iterator()
        iterator.seek(b"")

        try:
            while iterator.valid() and len(samples) < sample_size:
                if max_items is not None and processed >= max_items:
                    break

                # Only examine every skip_size-th item
                if processed % skip_size == 0:
                    key_bytes = iterator.key()

                    if not key_bytes.startswith(b'__'):
                        examined += 1

                        if return_keys:
                            samples.append((key_bytes, iterator.value()))
                        else:
                            samples.append(iterator.value())

                        if len(samples) >= next_progress:
                            percent_complete = (float(len(samples)) / sample_size) * 100
                            print(f"    Progress: {len(samples):,}/{sample_size:,} samples ({percent_complete:.1f}%)")
                            next_progress += progress_interval

                processed += 1
                iterator.next()

        finally:
            # Explicit iterator cleanup
            del iterator

    print(f"  Skip sampling complete: {len(samples):,} samples from {examined:,} examinations")
    return samples