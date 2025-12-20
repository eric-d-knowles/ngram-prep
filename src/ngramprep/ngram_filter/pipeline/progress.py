# ngram_filter/pipeline/progress.py
"""Progress tracking and reporting for the ngram filter pipeline."""

from __future__ import annotations

import multiprocessing as mp
import signal
import time
from dataclasses import dataclass, fields
from typing import Optional

from setproctitle import setproctitle


@dataclass
class Counters:
    """Shared counters for tracking pipeline progress across processes."""

    # Reader-side counters
    items_scanned: mp.Value
    items_filtered: mp.Value
    items_enqueued: mp.Value
    reader_done: mp.Value

    # Writer-side counters
    items_buffered: mp.Value
    items_written: mp.Value
    bytes_flushed: mp.Value
    opened_dbs: mp.Value
    closed_dbs: mp.Value


@dataclass
class ProgressSnapshot:
    """Immutable snapshot of progress counters at a point in time."""

    items_scanned: int
    items_filtered: int
    items_enqueued: int
    reader_done: int
    items_buffered: int
    items_written: int
    bytes_flushed: int
    opened_dbs: int
    closed_dbs: int
    timestamp: float

    @property
    def items_kept_ratio(self) -> float:
        """Ratio of items kept after filtering (0.0 to 1.0)."""
        if self.items_scanned == 0:
            return 0.0
        return self.items_enqueued / self.items_scanned

    @property
    def items_filtered_ratio(self) -> float:
        """Ratio of items filtered out (0.0 to 1.0)."""
        return 1.0 - self.items_kept_ratio

    def scanning_rate(self, elapsed_time: float) -> float:
        """Items scanned per second."""
        if elapsed_time <= 0:
            return 0.0
        return self.items_scanned / elapsed_time

    def writing_rate(self, elapsed_time: float) -> float:
        """Items written per second."""
        if elapsed_time <= 0:
            return 0.0
        return self.items_written / elapsed_time


def create_counters(ctx: mp.context.BaseContext) -> Counters:
    """
    Create shared counters for progress tracking.

    Args:
        ctx: Multiprocessing context for creating shared values

    Returns:
        Initialized Counters object with shared memory values
    """
    return Counters(
        # Reader counters
        items_scanned=ctx.Value("Q", 0),
        items_filtered=ctx.Value("Q", 0),
        items_enqueued=ctx.Value("Q", 0),
        reader_done=ctx.Value("Q", 0),

        # Writer counters
        items_buffered=ctx.Value("Q", 0),
        items_written=ctx.Value("Q", 0),
        bytes_flushed=ctx.Value("Q", 0),
        opened_dbs=ctx.Value("Q", 0),
        closed_dbs=ctx.Value("Q", 0),
    )


def increment_counter(counter: mp.Value, delta: int = 1) -> None:
    """
    Thread-safe increment of a shared counter.

    Args:
        counter: Shared multiprocessing Value to increment
        delta: Amount to increment by (default: 1)
    """
    with counter.get_lock():
        counter.value += delta


def read_counter(counter: mp.Value) -> int:
    """
    Read the current value of a shared counter.

    Args:
        counter: Shared multiprocessing Value to read

    Returns:
        Current counter value
    """
    return counter.value


def snapshot_counters(counters: Counters) -> ProgressSnapshot:
    """
    Create an immutable snapshot of all counters.

    Args:
        counters: Counters object to snapshot

    Returns:
        ProgressSnapshot with current values
    """
    return ProgressSnapshot(
        items_scanned=read_counter(counters.items_scanned),
        items_filtered=read_counter(counters.items_filtered),
        items_enqueued=read_counter(counters.items_enqueued),
        reader_done=read_counter(counters.reader_done),
        items_buffered=read_counter(counters.items_buffered),
        items_written=read_counter(counters.items_written),
        bytes_flushed=read_counter(counters.bytes_flushed),
        opened_dbs=read_counter(counters.opened_dbs),
        closed_dbs=read_counter(counters.closed_dbs),
        timestamp=time.perf_counter()
    )


def print_phase_banner() -> None:
    """Print the pipeline phase 1 banner and headers."""
    field_width = 14
    fields = ["items", "kept%", "workers", "units", "rate", "elapsed"]
    line = "─"
    print('\n' + ''.join(f"{field:^{field_width}}" for field in fields))
    print(''.join(f"{line*field_width:^{field_width}}" for field in fields))


class ProgressFormatter:
    """Formats progress statistics for display."""

    @staticmethod
    def format_count(count: int) -> str:
        """Format a count with K/M/B suffixes to 2 decimal places."""
        if count >= 1_000_000_000:
            return f"{count / 1_000_000_000:.2f}B"
        elif count >= 1_000_000:
            return f"{count / 1_000_000:.2f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.2f}K"
        else:
            return str(count)

    @staticmethod
    def format_rate(items_per_second: float) -> str:
        """Format a rate for display (e.g., '1.2k/s', '850/s')."""
        if items_per_second >= 1000:
            return f"{items_per_second / 1000:.1f}k/s"
        return f"{items_per_second:.0f}/s"

    @staticmethod
    def format_elapsed_time(seconds: float) -> str:
        """Format elapsed time for display."""
        if seconds >= 3600:  # Show hours for long runs
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h{minutes:02d}m"
        elif seconds >= 60:  # Show minutes for medium runs
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m{secs:02d}s"
        else:
            return f"{seconds:.0f}s"

    @staticmethod
    def format_readers_status(done: int, total: Optional[int] = None) -> str:
        """Format reader completion status."""
        if total is None:
            return str(done)
        return f"{done}/{total}"

    @classmethod
    def format_progress_line(
            cls,
            snapshot: ProgressSnapshot,
            start_time: float,
            work_progress=None,
            num_workers: Optional[int] = None
    ) -> str:
        """
        Format a complete progress line for display.

        Args:
            snapshot: Current progress snapshot
            start_time: Pipeline start time
            work_progress: WorkProgress with unit/worker info (optional)
            num_workers: Total number of workers (optional)

        Returns:
            Formatted progress line matching the header format
        """
        # Format elapsed time
        elapsed = max(1e-9, snapshot.timestamp - start_time)
        elapsed_str = cls.format_elapsed_time(elapsed)

        # Format items per second
        throughput_str = cls.format_rate(snapshot.scanning_rate(elapsed))

        # Format items scanned
        items_str = cls.format_count(snapshot.items_scanned)

        # Format kept percentage to 1 decimal place
        kept_pct = snapshot.items_kept_ratio * 100
        kept_str = f"{kept_pct:.1f}%"

        # Format worker info: active/total (or just active if total unknown)
        if work_progress and num_workers:
            active = work_progress.active_workers
            total = num_workers
            workers_str = f"{active}/{total}"
        elif work_progress:
            # No total count, just show active
            workers_str = f"{work_progress.active_workers}a"
        else:
            workers_str = "-"

        # Format units info: show stages pending→processing→completed
        if work_progress:
            pending = work_progress.pending
            processing = work_progress.processing
            completed = work_progress.completed

            # Show stages: pending·processing·completed
            units_str = f"{pending}·{processing}·{completed}"
        else:
            units_str = "-"

        fields = [items_str, kept_str, workers_str, units_str, throughput_str, elapsed_str]
        field_width = 14

        return ''.join(f"{field:^{field_width}}" for field in fields)


def run_progress_reporter(
        counters: Counters,
        start_time: float,
        update_interval: float = 1.0,
        stop_event: Optional[mp.Event] = None,
        num_workers: Optional[int] = None,
        work_tracker_path: Optional = None,
) -> None:
    """
    Run a progress reporter that periodically prints statistics.

    Args:
        counters: Shared counters to monitor
        start_time: Pipeline start time
        update_interval: Seconds between progress updates
        stop_event: Event to signal when to stop reporting
        num_workers: Total number of workers
        work_tracker_path: Path to work tracker database for unit/worker info
    """
    # Set process title for monitoring
    setproctitle("ngf:reporter")

    # Ignore interrupt signals to avoid interfering with main process
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # Initialize work tracker if path provided
    work_tracker = None
    if work_tracker_path:
        from ngramprep.tracking import WorkTracker
        work_tracker = WorkTracker(work_tracker_path)

    # Handle single-shot reporting (no stop event)
    if stop_event is None:
        snapshot = snapshot_counters(counters)
        work_progress = work_tracker.get_progress(num_workers=num_workers) if work_tracker else None
        progress_line = ProgressFormatter.format_progress_line(
            snapshot, start_time, work_progress, num_workers
        )
        print(progress_line, flush=True)
        return

    # Continuous reporting loop
    next_update = time.perf_counter() + max(0.0, update_interval)

    while not stop_event.is_set():
        current_time = time.perf_counter()

        if current_time >= next_update:
            snapshot = snapshot_counters(counters)
            work_progress = work_tracker.get_progress(num_workers=num_workers) if work_tracker else None
            progress_line = ProgressFormatter.format_progress_line(
                snapshot, start_time, work_progress, num_workers
            )
            print(progress_line, flush=True)
            next_update += max(0.0, update_interval)

        time.sleep(0.05)  # Small sleep to avoid busy waiting

    # Print final statistics
    final_snapshot = snapshot_counters(counters)
    final_work_progress = work_tracker.get_progress(num_workers=num_workers) if work_tracker else None
    final_progress = ProgressFormatter.format_progress_line(
        final_snapshot, start_time, final_work_progress, num_workers
    )
    print(final_progress, flush=True)