import signal
import sys
import time
import re
import struct
from collections import defaultdict

import rocks_shim as rs
from ngramprep.utilities.display import format_banner, truncate_path_to_fit
from ngramprep.utilities.progress import ProgressDisplay

DISPLAY_WIDTH = 100
_display = ProgressDisplay(width=DISPLAY_WIDTH)
COUNT_FIELD_WIDTH = 15


def _print_start_banner(db_path, progress_interval, grouping=None):
    db_line = truncate_path_to_fit(db_path, "Database: ", total_width=DISPLAY_WIDTH)
    print(format_banner("DATABASE ITEM COUNTER", width=DISPLAY_WIDTH, style="━"))
    print(db_line)
    print(f"Progress interval: every {progress_interval:,} items")
    if grouping:
        print(f"Grouping by: {grouping}")
    print()
    print(format_banner("COUNTING", width=DISPLAY_WIDTH, style="─"), flush=True)


def _print_progress(count, elapsed):
    rate_str = ProgressDisplay.format_rate(count, elapsed, "items")
    print(f"[{count:>{COUNT_FIELD_WIDTH},}] | elapsed {elapsed:8.1f}s | rate {rate_str}", flush=True)


def _print_summary(db_path, count, total_time, group_counts=None):
    rate_str = ProgressDisplay.format_rate(count, total_time, "items")
    # content area inside box is 94 chars (100 width minus borders/padding)
    db_line = truncate_path_to_fit(db_path, "Database: ", total_width=94)
    summary_items = {
        "Items": f"{count:,}",
        "Elapsed": f"{total_time:.2f}s",
        "Avg rate": rate_str,
        "Database": db_line,
    }
    
    if group_counts:
        summary_items["Groups"] = f"{len(group_counts):,}"
    
    print()
    _display.print_summary_box(
        title="COUNT COMPLETE",
        items=summary_items,
        box_width=DISPLAY_WIDTH,
    )
    
    if group_counts:
        print()
        print(format_banner("GROUP COUNTS", width=DISPLAY_WIDTH, style="─"))
        for group in sorted(group_counts.keys()):
            print(f"  {group:20s}: {group_counts[group]:>12,} items")
        print()


cdef bytes _extract_year_bin(bytes key):
    """
    Extract a year/bin from a pivoted key.

    Pivoted DB keys are stored as a 4-byte big-endian integer for the year,
    followed by the ngram text. We decode that binary prefix.
    """
    cdef Py_ssize_t key_len = len(key)
    cdef unsigned int year
    cdef const unsigned char* key_ptr
    
    # Fast path: 4-byte big-endian year prefix used by pivoted DBs
    if key_len >= 4:
        key_ptr = <const unsigned char*>(<char*>key)
        # Manually unpack big-endian 4-byte unsigned int
        year = ((key_ptr[0] << 24) | 
                (key_ptr[1] << 16) | 
                (key_ptr[2] << 8) | 
                key_ptr[3])
        
        # Sanity check: years should be in reasonable range
        if 0 < year < 10000:
            return str(year).encode('utf-8')
    
    return b"unknown"


def count_db_items(db_path, long progress_interval = 10_000_000, grouping=None):
    """
    Count items in a RocksDB database, optionally grouped by key prefix.
    
    Args:
        db_path: Path to the RocksDB database
        progress_interval: Print progress every N items (default: 10,000,000)
        grouping: Optional grouping specification. Can be:
            - None: Return total count (default behavior)
            - 'year_bin': Fast extraction of [YEAR] prefix (recommended for pivoted DBs)
            - str: Regex pattern to extract group from key (e.g., r'^\[(\d+)\]' for year bins)
            - callable: Function that takes a key (bytes) and returns a group identifier (str)
            
    Returns:
        int if grouping is None, dict[str, int] if grouping is specified
        
    Examples:
        # Total count
        >>> count_db_items('/path/to/db')
        1234567
        
        # Count by year bin (fast path)
        >>> count_db_items('/path/to/db', grouping='year_bin')
        {'1900': 45123, '1905': 46782, '1910': 47234, ...}
        
        # Count by year bin (regex - slower)
        >>> count_db_items('/path/to/db', grouping=r'^\[(\d+)\]')
        {'1900': 45123, '1905': 46782, '1910': 47234, ...}
        
        # Custom grouping function
        >>> count_db_items('/path/to/db', grouping=lambda k: k[:4].decode('utf-8', 'replace'))
        {'[190': 45123, '[191': 46782, ...}
    """
    cdef double start_time, elapsed, end_time, total_time
    cdef long count = 0
    cdef long next_progress = progress_interval
    cdef bytes key
    cdef bytes group_bytes
    
    # Prepare grouping function
    group_counts = None
    grouping_fn = None
    use_fast_year_bin = False
    
    if grouping is not None:
        group_counts = defaultdict(int)
        
        if grouping == 'year_bin':
            # Use optimized Cython function for year bins
            use_fast_year_bin = True
        elif isinstance(grouping, str):
            # Compile regex pattern
            pattern = re.compile(grouping.encode('utf-8') if isinstance(grouping, str) else grouping)
            def grouping_fn(key):
                match = pattern.search(key)
                if match:
                    return match.group(1).decode('utf-8', 'replace') if match.lastindex else match.group(0).decode('utf-8', 'replace')
                return "unknown"
        elif callable(grouping):
            grouping_fn = grouping
        else:
            raise ValueError(f"grouping must be None, 'year_bin', str (regex), or callable, got {type(grouping)}")

    def signal_handler(signum, frame):
        print(f"\nInterrupt received at count {count:,}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    _print_start_banner(db_path, progress_interval, grouping)

    try:
        with rs.open(db_path, mode="r") as db:
            start_time = time.perf_counter()
            iterator = db.iterator()
            iterator.seek(b"")

            # Fast path: no grouping
            if grouping is None:
                while iterator.valid():
                    count += 1
                    
                    if count >= next_progress:
                        elapsed = time.perf_counter() - start_time
                        _print_progress(count, elapsed)
                        next_progress += progress_interval
                    
                    iterator.next()
            
            # Fast path: year_bin extraction
            elif use_fast_year_bin:
                while iterator.valid():
                    count += 1
                    key = iterator.key()
                    group_bytes = _extract_year_bin(key)
                    group_counts[group_bytes.decode('utf-8', 'replace')] += 1
                    
                    if count >= next_progress:
                        elapsed = time.perf_counter() - start_time
                        _print_progress(count, elapsed)
                        next_progress += progress_interval
                    
                    iterator.next()
            
            # Slow path: custom grouping function
            else:
                while iterator.valid():
                    count += 1
                    key = iterator.key()
                    group = grouping_fn(key)
                    group_counts[group] += 1

                    if count >= next_progress:
                        elapsed = time.perf_counter() - start_time
                        _print_progress(count, elapsed)
                        next_progress += progress_interval

                    iterator.next()

            end_time = time.perf_counter()
            total_time = end_time - start_time

            del iterator

            _print_summary(db_path, count, total_time, group_counts)

    except Exception as e:
        print(f"Error at count {count}: {e}")
        raise

    # Return counts or group_counts
    if group_counts is not None:
        return dict(group_counts)
    return count