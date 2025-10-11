# parallel/split_monitor.py
"""Dynamic work unit splitting monitor for load balancing."""

from __future__ import annotations

import multiprocessing as mp
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from setproctitle import setproctitle

from ngram_prep.parallel.work_tracker import WorkTracker

__all__ = ["split_monitor", "SplitMonitorConfig"]


@dataclass
class SplitMonitorConfig:
    """Configuration for split monitor behavior."""

    check_interval: float = 1.0
    """Time between checks (seconds)"""

    starvation_threshold: int = 3
    """Number of consecutive checks below threshold before splitting"""

    log_file: Optional[Path] = None
    """Optional path to log file for split monitor activity"""


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

    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    logger.handlers.clear()

    if config.log_file:
        # File handler with detailed format
        fh = logging.FileHandler(config.log_file, mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("=" * 80)
        logger.info("Split monitor started")
        logger.info(f"Configuration: check_interval={config.check_interval}s, "
                   f"starvation_threshold={config.starvation_threshold}")
        logger.info("=" * 80)

    work_tracker = WorkTracker(work_tracker_path)

    # Track consecutive checks below threshold
    starvation_checks = 0
    cycle_count = 0

    while not stop_event.is_set():
        try:
            cycle_count += 1
            progress = work_tracker.get_progress(num_workers=num_workers)

            # Track how long workers have been idle
            workers_idle = progress.processing > 0 and progress.idle_workers > 0

            if workers_idle:
                starvation_checks += 1
            else:
                starvation_checks = 0

            # Log check cycle
            if config.log_file:
                logger.debug(
                    f"[Cycle {cycle_count}] Check - "
                    f"Workers: {num_workers} total, {progress.processing} processing, "
                    f"{progress.idle_workers} idle | "
                    f"Units: {progress.pending} pending, {progress.completed} completed | "
                    f"Starvation checks: {starvation_checks}/{config.starvation_threshold}"
                )

            # Workers are "starving" after being idle for threshold consecutive checks
            starving_count = progress.idle_workers if starvation_checks >= config.starvation_threshold else 0
            work_tracker.set_starving_workers(starving_count)

            # Log starvation confirmation
            if config.log_file and starving_count > 0:
                logger.info(
                    f"[Cycle {cycle_count}] STARVATION CONFIRMED - "
                    f"{starving_count} workers starving (threshold reached: "
                    f"{starvation_checks} >= {config.starvation_threshold})"
                )

            # Split when workers are starving
            if starving_count > 0:
                # Split enough processing units to feed all starving workers
                # Each split creates 2 child units, so we need fewer splits than starving workers
                num_to_split = (starving_count + 1) // 2

                if config.log_file:
                    logger.info(
                        f"[Cycle {cycle_count}] ATTEMPTING SPLITS - "
                        f"Target: {num_to_split} splits needed to feed {starving_count} starving workers"
                    )

                splits_done = 0

                # Get all candidate units (processing first, then pending)
                candidate_units = work_tracker.get_all_splittable_units(prefer_processing=True)

                if config.log_file:
                    logger.debug(
                        f"[Cycle {cycle_count}] Found {len(candidate_units)} candidate units for splitting"
                    )

                # Try splitting candidates until we get enough splits
                failed_attempts = 0
                failed_reasons = []
                for unit_id in candidate_units:
                    if splits_done >= num_to_split:
                        break

                    try:
                        work_tracker.split_work_unit(unit_id)
                        splits_done += 1

                        if config.log_file:
                            logger.info(
                                f"[Cycle {cycle_count}] SPLIT SUCCESS - "
                                f"Unit {unit_id} split successfully ({splits_done}/{num_to_split} complete)"
                            )

                        # Note: Parent worker will detect the split status and clean up
                        # its own output after finishing (see worker.py lines 100-104)
                    except ValueError as e:
                        # Can't split this unit further - try next one
                        failed_attempts += 1
                        if failed_attempts <= 3:  # Keep first 3 failure reasons
                            failed_reasons.append(f"{unit_id}: {str(e)}")

                        if config.log_file:
                            logger.debug(
                                f"[Cycle {cycle_count}] SPLIT FAILED - "
                                f"Unit {unit_id} cannot be split: {str(e)}"
                            )
                        continue

                # Log split results summary
                if config.log_file:
                    if splits_done > 0:
                        logger.info(
                            f"[Cycle {cycle_count}] SPLIT ROUND COMPLETE - "
                            f"Successfully split {splits_done}/{num_to_split} units "
                            f"(attempted {len(candidate_units)} candidates, "
                            f"{failed_attempts} failed)"
                        )
                    else:
                        logger.warning(
                            f"[Cycle {cycle_count}] SPLIT ROUND FAILED - "
                            f"Could not split any units (tried {failed_attempts} candidates). "
                            f"Reasons: {'; '.join(failed_reasons[:3]) if failed_reasons else 'none'}"
                        )

                # Debug: Report if we couldn't split enough units
                if starving_count > 0 and splits_done == 0 and message_queue:
                    reasons_str = "; ".join(failed_reasons) if failed_reasons else "unknown"
                    message_queue.put(
                        f"[split monitor] Warning: {starving_count} starving workers but "
                        f"couldn't split any units (tried {failed_attempts} units). "
                        f"Sample failures: {reasons_str}"
                    )

                if splits_done > 0:
                    starvation_checks = 0
                    if config.log_file:
                        logger.debug(
                            f"[Cycle {cycle_count}] Starvation checks reset to 0 after successful splits"
                        )

            time.sleep(config.check_interval)

        except Exception as e:
            if config.log_file:
                logger.error(f"[Cycle {cycle_count}] ERROR in split monitor: {e}", exc_info=True)
            if message_queue:
                message_queue.put(f"[split monitor] Error: {e}")
            time.sleep(5)

    # Log shutdown
    if config.log_file:
        logger.info("=" * 80)
        logger.info(f"Split monitor stopping after {cycle_count} cycles")
        logger.info("=" * 80)
