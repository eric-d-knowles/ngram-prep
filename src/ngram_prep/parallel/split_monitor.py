# parallel/split_monitor.py
"""Dynamic work unit splitting monitor for load balancing."""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from setproctitle import setproctitle

from ngram_prep.parallel.work_tracker import WorkTracker

__all__ = ["split_monitor", "SplitMonitorConfig"]


@dataclass
class SplitMonitorConfig:
    """Configuration for split monitor behavior."""

    check_interval: float = 0.5
    """Time between checks (seconds)"""

    starvation_threshold: int = 3
    """Number of consecutive checks below threshold before splitting"""


def split_monitor(
        work_tracker_path: Path,
        num_workers: int,
        stop_event: mp.Event,
        output_manager: Optional[object] = None,
        message_queue: Optional[mp.Queue] = None,
        config: Optional[SplitMonitorConfig] = None,
) -> None:
    """
    Monitor work queue and split units when workers are idle.

    This function runs in a separate process and continuously monitors the work tracker
    to detect when workers are starving (i.e., there aren't enough pending work units
    to keep all workers busy). When starvation is detected for multiple consecutive
    checks, it automatically splits a work unit to create more parallelism.

    The monitor uses a conservative approach:
    - Workers are considered "starving" when processing units < total workers
    - Splitting only occurs after STARVATION_THRESHOLD consecutive starved checks
    - This prevents excessive splitting while ensuring good load balance
    - Workers detect splits and clean up their own partial outputs

    Args:
        work_tracker_path: Path to work tracker database
        num_workers: Total number of worker processes
        stop_event: Event to signal monitor to stop
        output_manager: Optional output manager (currently unused, kept for compatibility)
        message_queue: Optional queue for sending status messages
        config: Optional configuration (uses defaults if None)

    Example:
        >>> stop_event = mp.Event()
        >>> monitor_proc = mp.Process(
        ...     target=split_monitor,
        ...     args=(tracker_path, 8, stop_event)
        ... )
        >>> monitor_proc.start()
        >>> # ... workers run ...
        >>> stop_event.set()
        >>> monitor_proc.join()
    """
    if config is None:
        config = SplitMonitorConfig()

    # Set process title for monitoring
    setproctitle("ngf:split-monitor")

    work_tracker = WorkTracker(work_tracker_path)

    # Track consecutive checks below threshold
    starvation_checks = 0

    while not stop_event.is_set():
        try:
            progress = work_tracker.get_progress()

            # Workers are starving if we don't have enough pending buffer
            # Maintain pending >= min(num_workers, processing) to ensure workers never idle
            # This is proactive: we create work BEFORE workers go idle
            has_work = progress.processing > 0 or progress.pending > 0
            target_pending = min(num_workers, max(progress.processing, 1))
            is_starving = progress.pending < target_pending

            if has_work and is_starving:
                starvation_checks += 1
            else:
                starvation_checks = 0

            # Split after persistent starvation
            if starvation_checks >= config.starvation_threshold:
                unit_id = work_tracker.get_any_splittable_unit()

                if unit_id:
                    try:
                        work_tracker.split_work_unit(unit_id)
                        starvation_checks = 0

                        # Note: Parent worker will detect the split status and clean up
                        # its own output after finishing (see worker.py lines 100-104)
                    except ValueError as e:
                        # Can't split further - accept some idle workers
                        pass

            time.sleep(config.check_interval)

        except Exception as e:
            if message_queue:
                message_queue.put(f"[split monitor] Error: {e}")
            time.sleep(5)
