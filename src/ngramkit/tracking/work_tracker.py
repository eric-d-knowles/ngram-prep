# ngram_filter/tracking/work_tracker.py
"""High-level API for work unit tracking with work-stealing over key ranges."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .types import WorkUnit, WorkProgress
from .database import init_tracking_database
from .work_unit_store import WorkUnitStore
from .progress_tracker import ProgressTracker

__all__ = ["WorkTracker", "WorkUnit", "WorkProgress"]


class WorkTracker:
    """
    High-level API for tracking work units in parallel execution.

    This class provides a unified interface that delegates to specialized components:
    - Database schema management
    - CRUD operations on work units
    - Progress tracking and statistics
    """

    def __init__(self, db_path: Path, claim_order: str = "sequential"):
        """
        Initialize work tracker.

        Args:
            db_path: Path to SQLite database for tracking
            claim_order: Order for claiming work units - "sequential" or "random"
        """
        self.db_path = Path(db_path)

        # Initialize database schema
        init_tracking_database(self.db_path)

        # Initialize specialized components
        self._store = WorkUnitStore(self.db_path, claim_order=claim_order)
        self._progress = ProgressTracker(self.db_path)

    # =========================================================================
    # Work Unit Lifecycle Methods
    # =========================================================================

    def add_work_units(self, work_units: List[WorkUnit]) -> None:
        """
        Add work units to the tracker.

        Args:
            work_units: List of work units to add
        """
        self._store.add_work_units(work_units)

    def claim_work_unit(self, worker_id: str, max_retries: int = 5) -> Optional[WorkUnit]:
        """
        Atomically claim the next available work unit.

        Args:
            worker_id: Identifier for the worker claiming the unit
            max_retries: Maximum number of retry attempts for database locks

        Returns:
            The claimed WorkUnit or None if no work is available
        """
        return self._store.claim_work_unit(worker_id, max_retries)

    def get_work_unit(self, unit_id: str) -> Optional[WorkUnit]:
        """
        Get a work unit by ID.

        Args:
            unit_id: ID of the work unit

        Returns:
            WorkUnit or None if not found
        """
        return self._store.get_work_unit(unit_id)

    def checkpoint_position(self, unit_id: str, position: bytes, max_retries: int = 5) -> None:
        """
        Update the current scan position for a work unit.

        Args:
            unit_id: ID of work unit
            position: Current scan position (key)
            max_retries: Maximum number of retry attempts for database locks
        """
        self._store.checkpoint_position(unit_id, position, max_retries)

    def fail_work_unit(self, unit_id: str, max_retries: int = 5) -> None:
        """
        Mark a work unit as failed.

        Args:
            unit_id: ID of failed work unit
            max_retries: Maximum number of retry attempts for database locks
        """
        self._store.fail_work_unit(unit_id, max_retries)

    def complete_work_unit(self, unit_id: str, max_retries: int = 5) -> None:
        """
        Mark a work unit as completed (finished filtering, ready for ingest).

        Transitions from 'processing' to 'completed'.

        Args:
            unit_id: ID of work unit to mark as completed
            max_retries: Maximum number of retry attempts for database locks
        """
        self._store.complete_work_unit(unit_id, max_retries)

    def get_next_completed_unit(self, max_retries: int = 5) -> Optional[str]:
        """
        Get next completed-but-not-ingested unit for ingestion.

        WARNING: This is NOT atomic - use claim_completed_unit_for_ingest() instead.

        Returns:
            Unit ID of next completed unit, or None if none available
        """
        return self._store.get_next_completed_unit(max_retries)

    def claim_completed_unit_for_ingest(self, max_retries: int = 5) -> Optional[str]:
        """
        Atomically claim next completed unit for ingestion.

        Returns:
            Unit ID that was claimed, or None if no completed units available
        """
        return self._store.claim_completed_unit_for_ingest(max_retries)

    def ingest_work_unit(self, unit_id: str, max_retries: int = 5) -> None:
        """
        Mark a work unit as ingested (successfully added to destination DB).

        Transitions from 'processing' or 'completed' to 'ingested'.

        Args:
            unit_id: ID of work unit to mark as ingested
            max_retries: Maximum number of retry attempts for database locks
        """
        self._store.ingest_work_unit(unit_id, max_retries)

    # =========================================================================
    # Progress Tracking Methods
    # =========================================================================

    def get_progress(self, num_workers: Optional[int] = None) -> WorkProgress:
        """
        Get current progress statistics.

        Args:
            num_workers: Total number of workers (optional, used to calculate idle_workers)

        Returns:
            WorkProgress with current counts and active worker count
        """
        return self._progress.get_progress(num_workers)

    # =========================================================================
    # Maintenance Methods
    # =========================================================================

    def reset_incomplete_ingestions(self, output_dir: Path) -> int:
        """
        Reset units marked as 'ingesting' or 'ingested' that still have shards on disk.

        This recovery method should be called at pipeline startup to recover from
        interrupted ingestion operations.

        Args:
            output_dir: Directory containing shard databases

        Returns:
            Number of units reset
        """
        return self._store.reset_incomplete_ingestions(output_dir)

    def clear_all_work_units(self) -> None:
        """Clear all work units from the tracker."""
        self._store.clear_all_work_units()

    def reset_all_processing_units(self) -> int:
        """
        Reset all processing units back to pending on restart.

        Units that have children (were split) remain 'completed'.
        Units without children are reset to 'pending'.

        Returns:
            Number of units reset
        """
        return self._store.reset_all_processing_units()

    def reset_all_work_units(self) -> int:
        """
        Reset ALL work units (including completed ones) back to pending.

        This is used in reprocess mode to re-run all work from scratch
        while preserving the work unit partitions/boundaries.

        Returns:
            Number of units reset
        """
        return self._store.reset_all_work_units()
