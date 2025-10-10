# parallel/work_tracker.py
"""Work unit tracking for parallel execution with work-stealing over key ranges."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional

from ngram_prep.parallel.types import WorkUnit, WorkProgress

__all__ = ["WorkTracker", "WorkUnit", "WorkProgress"]


class WorkTracker:
    """Tracks work unit status using SQLite."""

    def __init__(self, db_path: Path):
        """
        Initialize work tracker.

        Args:
            db_path: Path to SQLite database for tracking
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the tracking database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS work_units
                         (
                             unit_id TEXT PRIMARY KEY,
                             start_key BLOB,
                             end_key BLOB,
                             status TEXT DEFAULT 'pending',
                             claimed_by TEXT,
                             claimed_at REAL,
                             completed_at REAL,
                             parent_id TEXT,
                             current_position BLOB
                         )
                         """)
            conn.execute("""
                         CREATE INDEX IF NOT EXISTS idx_status
                             ON work_units(status)
                         """)
            conn.commit()

    def add_work_units(self, work_units: List[WorkUnit]) -> None:
        """
        Add work units to the tracker.

        Args:
            work_units: List of work units to add
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            for unit in work_units:
                # Handle both old WorkUnit (without parent_id) and new (with parent_id)
                parent_id = getattr(unit, 'parent_id', None)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO work_units 
                    (unit_id, start_key, end_key, status, parent_id)
                    VALUES (?, ?, ?, 'pending', ?)
                    """,
                    (
                        unit.unit_id,
                        unit.start_key,
                        unit.end_key,
                        parent_id,
                    )
                )
            conn.commit()

    def claim_work_unit(
            self,
            worker_id: str,
            max_retries: int = 5
    ) -> Optional[WorkUnit]:
        """
        Atomically claim the next available work unit.

        Args:
            worker_id: Identifier for the worker claiming the unit
            max_retries: Maximum number of retry attempts for database locks

        Returns:
            The claimed WorkUnit or None if no work is available
        """
        import time

        with sqlite3.connect(str(self.db_path)) as conn:
            for attempt in range(max_retries):
                try:
                    cursor = conn.execute(
                        """
                        UPDATE work_units
                        SET status     = 'processing',
                            claimed_by = ?,
                            claimed_at = ?
                        WHERE unit_id = (
                            SELECT unit_id
                            FROM work_units
                            WHERE status = 'pending'
                            ORDER BY unit_id
                            LIMIT 1
                            )
                            RETURNING unit_id, start_key, end_key, parent_id, current_position
                        """,
                        (worker_id, time.time())
                    )

                    row = cursor.fetchone()
                    if row:
                        return WorkUnit(
                            unit_id=row[0],
                            start_key=row[1],
                            end_key=row[2],
                            parent_id=row[3],
                            current_position=row[4],
                        )
                    return None  # No work available

                except sqlite3.OperationalError as e:
                    if "locked" in str(e) and attempt < max_retries - 1:
                        time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                        continue
                    raise

            return None  # All retries exhausted

    def checkpoint_position(self, unit_id: str, position: bytes, max_retries: int = 5) -> None:
        """
        Update the current scan position for a work unit.

        Args:
            unit_id: ID of work unit
            position: Current scan position (key)
            max_retries: Maximum number of retry attempts for database locks
        """
        import time

        for attempt in range(max_retries):
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute(
                        """
                        UPDATE work_units
                        SET current_position = ?
                        WHERE unit_id = ?
                        """,
                        (position, unit_id)
                    )
                    conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    continue
                raise

    def complete_work_unit(self, unit_id: str, max_retries: int = 5) -> None:
        """
        Mark a work unit as completed.

        Args:
            unit_id: ID of completed work unit
            max_retries: Maximum number of retry attempts for database locks
        """
        import time

        for attempt in range(max_retries):
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute(
                        """
                        UPDATE work_units
                        SET status       = 'completed',
                            completed_at = ?
                        WHERE unit_id = ?
                        """,
                        (time.time(), unit_id)
                    )
                    conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    continue
                raise

    def fail_work_unit(self, unit_id: str, max_retries: int = 5) -> None:
        """
        Mark a work unit as failed.

        Args:
            unit_id: ID of failed work unit
            max_retries: Maximum number of retry attempts for database locks
        """
        import time

        for attempt in range(max_retries):
            try:
                with sqlite3.connect(str(self.db_path)) as conn:
                    conn.execute(
                        """
                        UPDATE work_units
                        SET status = 'failed'
                        WHERE unit_id = ?
                        """,
                        (unit_id,)
                    )
                    conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    continue
                raise

    def get_progress(self) -> WorkProgress:
        """
        Get current progress statistics.

        Returns:
            WorkProgress with current counts
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT status, COUNT(*) as count
                FROM work_units
                GROUP BY status
                """
            )

            counts = {row[0]: row[1] for row in cursor}
            total = sum(counts.values())

            return WorkProgress(
                total=total,
                pending=counts.get('pending', 0),
                processing=counts.get('processing', 0),
                completed=counts.get('completed', 0),
                failed=counts.get('failed', 0),
                split=counts.get('split', 0),
            )

    def clear_all_work_units(self) -> None:
        """Clear all work units from the tracker."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM work_units")
            conn.commit()

    def reset_all_processing_units(self) -> int:
        """
        Reset all processing units back to pending.

        Units that were split (have children) are marked as 'split' instead of 'pending'.

        Returns:
            Number of units reset
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            # First, mark any processing units that have children as split
            # (these were split but the status update didn't complete before crash)
            conn.execute(
                """
                UPDATE work_units
                SET status = 'split'
                WHERE status = 'processing'
                  AND unit_id IN (
                    SELECT DISTINCT parent_id
                    FROM work_units
                    WHERE parent_id IS NOT NULL
                )
                """
            )

            # Then reset remaining processing units to pending
            # (these are either original units or children that were interrupted)
            cursor = conn.execute(
                """
                UPDATE work_units
                SET status     = 'pending',
                    claimed_by = NULL,
                    claimed_at = NULL
                WHERE status = 'processing'
                """
            )
            conn.commit()
            return cursor.rowcount

    def split_work_unit(self, unit_id: str) -> tuple[WorkUnit, WorkUnit]:
        """
        Split a work unit into two halves.

        NOTE: If the unit is currently being processed, the worker will fail
        gracefully when trying to complete it (status changed to 'split').
        Any partial outputs from the parent unit should be cleaned up by the caller
        after the split completes.

        Args:
            unit_id: ID of unit to split

        Returns:
            Tuple of (left_unit, right_unit)

        Raises:
            ValueError: If unit cannot be split
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                "SELECT * FROM work_units WHERE unit_id = ?",
                (unit_id,)
            )
            row = cursor.fetchone()

            if not row:
                raise ValueError(f"Unit {unit_id} not found")

            if row['status'] not in ('pending', 'processing'):
                raise ValueError(f"Unit {unit_id} status is {row['status']}, cannot split")

            start_key = row['start_key']
            end_key = row['end_key']

            # Find midpoint key using partitioning module
            from ngram_prep.parallel.partitioning import find_midpoint_key
            midpoint = find_midpoint_key(start_key, end_key)

            if midpoint is None or midpoint == start_key or midpoint == end_key:
                raise ValueError(f"Unit {unit_id} cannot be split further")

            # Create two new units
            left_id = f"{unit_id}_L"
            right_id = f"{unit_id}_R"

            # Mark original unit as split
            conn.execute(
                "UPDATE work_units SET status = 'split' WHERE unit_id = ?",
                (unit_id,)
            )

            # Insert new units as pending
            conn.execute(
                """
                INSERT INTO work_units
                    (unit_id, start_key, end_key, status, parent_id)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (left_id, start_key, midpoint, unit_id)
            )

            conn.execute(
                """
                INSERT INTO work_units
                    (unit_id, start_key, end_key, status, parent_id)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (right_id, midpoint, end_key, unit_id)
            )

            conn.commit()

        # NOTE: Caller is responsible for handling any partial outputs from the parent unit

        return (
            WorkUnit(left_id, start_key, midpoint, parent_id=unit_id),
            WorkUnit(right_id, midpoint, end_key, parent_id=unit_id)
        )

    def get_any_splittable_unit(self) -> Optional[str]:
        """Get any pending or processing work unit that can be split."""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Prefer processing units (likely larger), then original units
            cursor = conn.execute(
                """
                SELECT unit_id
                FROM work_units
                WHERE status IN ('processing', 'pending')
                ORDER BY
                    CASE status WHEN 'processing' THEN 0 ELSE 1 END,
                    CASE WHEN parent_id IS NULL THEN 0 ELSE 1 END,
                    RANDOM()
                    LIMIT 1
                """
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def get_unit_status(self, unit_id: str) -> Optional[str]:
        """
        Get the current status of a work unit.

        Args:
            unit_id: ID of the work unit

        Returns:
            Status string or None if unit not found
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "SELECT status FROM work_units WHERE unit_id = ?",
                (unit_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def get_failed_unit_ids(self) -> list[str]:
        """
        Get list of unit IDs that have failed (but not split).

        These units have partial outputs that can be safely cleaned up.

        Returns:
            List of unit IDs with status 'failed'
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                """
                SELECT unit_id
                FROM work_units
                WHERE status = 'failed'
                """
            )
            return [row[0] for row in cursor]