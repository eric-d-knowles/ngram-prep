import signal
import sys
import time
import rocks_shim as rs

def count_db_items(db_path, long progress_interval = 10_000_000):
    cdef double start_time, elapsed, rate, end_time, total_time
    cdef long count = 0
    cdef long next_progress = progress_interval

    def signal_handler(signum, frame):
        print(f"\nInterrupt received at count {count}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Database Item Counter")
    print("=" * 65)

    try:
        with rs.open(db_path, mode="r") as db:
            start_time = time.perf_counter()
            iterator = db.iterator()
            iterator.seek(b"")

            while iterator.valid():
                count += 1

                if count >= next_progress:
                    elapsed = time.perf_counter() - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    print(f"Progress: {count:,} items | {elapsed:.1f}s elapsed | {rate:,.0f} items/sec", flush=True)
                    next_progress += progress_interval

                iterator.next()

            end_time = time.perf_counter()
            total_time = end_time - start_time

            del iterator

            print("=" * 65)
            print(f"FINAL COUNT: {count:,} items")
            print(f"Total Time:  {total_time:.2f} seconds")
            print("=" * 65)

    except Exception as e:
        print(f"Error at count {count}: {e}")
        raise

    return count