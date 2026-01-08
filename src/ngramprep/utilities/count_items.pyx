import signal
import sys
import time

import rocks_shim as rs
from ngramprep.utilities.display import format_banner, truncate_path_to_fit
from ngramprep.utilities.progress import ProgressDisplay

DISPLAY_WIDTH = 100
_display = ProgressDisplay(width=DISPLAY_WIDTH)
COUNT_FIELD_WIDTH = 15


def _print_start_banner(db_path, progress_interval):
    db_line = truncate_path_to_fit(db_path, "Database: ", total_width=DISPLAY_WIDTH)
    print(format_banner("DATABASE ITEM COUNTER", width=DISPLAY_WIDTH, style="━"))
    print(db_line)
    print(f"Progress interval: every {progress_interval:,} items")
    print()
    print(format_banner("COUNTING", width=DISPLAY_WIDTH, style="─"), flush=True)


def _print_progress(count, elapsed):
    rate_str = ProgressDisplay.format_rate(count, elapsed, "items")
    print(f"[{count:>{COUNT_FIELD_WIDTH},}] | elapsed {elapsed:8.1f}s | rate {rate_str}", flush=True)


def _print_summary(db_path, count, total_time):
    rate_str = ProgressDisplay.format_rate(count, total_time, "items")
    # content area inside box is 94 chars (100 width minus borders/padding)
    db_line = truncate_path_to_fit(db_path, "Database: ", total_width=94)
    summary_items = {
        "Items": f"{count:,}",
        "Elapsed": f"{total_time:.2f}s",
        "Avg rate": rate_str,
        "Database": db_line,
    }
    print()
    _display.print_summary_box(
        title="COUNT COMPLETE",
        items=summary_items,
        box_width=DISPLAY_WIDTH,
    )


def count_db_items(db_path, long progress_interval = 10_000_000):
    cdef double start_time, elapsed, end_time, total_time
    cdef long count = 0
    cdef long next_progress = progress_interval

    def signal_handler(signum, frame):
        print(f"\nInterrupt received at count {count:,}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    _print_start_banner(db_path, progress_interval)

    try:
        with rs.open(db_path, mode="r") as db:
            start_time = time.perf_counter()
            iterator = db.iterator()
            iterator.seek(b"")

            while iterator.valid():
                count += 1

                if count >= next_progress:
                    elapsed = time.perf_counter() - start_time
                    _print_progress(count, elapsed)
                    next_progress += progress_interval

                iterator.next()

            end_time = time.perf_counter()
            total_time = end_time - start_time

            del iterator

            _print_summary(db_path, count, total_time)

    except Exception as e:
        print(f"Error at count {count}: {e}")
        raise

    return count