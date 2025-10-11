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
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
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
            # Metadata table for split monitor state
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS metadata
                         (
                             key TEXT PRIMARY KEY,
                             value TEXT
                         )
                         """)
            conn.commit()

    def add_work_units(self, work_units: List[WorkUnit]) -> None:
        """
        Add work units to the tracker.

        Args:
            work_units: List of work units to add
        """
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
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

        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
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
                with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
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

        Only updates if the unit is currently in 'processing' state, to prevent
        race conditions with split operations.

        Args:
            unit_id: ID of completed work unit
            max_retries: Maximum number of retry attempts for database locks

        Raises:
            ValueError: If unit is not in processing state (may have been split)
        """
        import time

        for attempt in range(max_retries):
            try:
                with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                    cursor = conn.execute(
                        """
                        UPDATE work_units
                        SET status       = 'completed',
                            completed_at = ?
                        WHERE unit_id = ? AND status = 'processing'
                        """,
                        (time.time(), unit_id)
                    )

                    if cursor.rowcount == 0:
                        # Check if unit was split or is in another state
                        cursor = conn.execute(
                            "SELECT status FROM work_units WHERE unit_id = ?",
                            (unit_id,)
                        )
                        row = cursor.fetchone()
                        if row:
                            current_status = row[0]
                            if current_status == 'completed':
                                # Already completed (by split with progress) - this is OK
                                conn.commit()
                                return
                            else:
                                raise ValueError(
                                    f"Cannot complete unit {unit_id}: "
                                    f"status is '{current_status}' (expected 'processing')"
                                )
                        else:
                            raise ValueError(f"Unit {unit_id} not found")

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
                with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
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

    def get_progress(self, num_workers: Optional[int] = None) -> WorkProgress:
        """
        Get current progress statistics.

        Args:
            num_workers: Total number of workers (optional, used to calculate idle_workers)

        Returns:
            WorkProgress with current counts and active worker count
        """
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
            # Count units that represent active/actual work:
            # - Leaf units (units not split): all statuses
            # - Parent units with status='completed' (split with progress preserved)
            # Excludes: Parent units with status='split' (split without progress, work discarded)
            cursor = conn.execute(
                """
                SELECT status, COUNT(*) as count
                FROM work_units
                WHERE (
                    -- Include leaf units (not yet split)
                    unit_id NOT IN (
                        SELECT DISTINCT parent_id
                        FROM work_units
                        WHERE parent_id IS NOT NULL
                    )
                    OR
                    -- Include parent units that completed with progress saved
                    (status = 'completed' AND unit_id IN (
                        SELECT DISTINCT parent_id
                        FROM work_units
                        WHERE parent_id IS NOT NULL
                    ))
                )
                GROUP BY status
                """
            )

            counts = {row[0]: row[1] for row in cursor}
            total = sum(counts.values())

            # Count distinct workers actively processing units
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT claimed_by)
                FROM work_units
                WHERE status = 'processing' AND claimed_by IS NOT NULL
                """
            )
            active_workers = cursor.fetchone()[0]

            # Calculate idle workers if total worker count provided
            idle_workers = (num_workers - active_workers) if num_workers is not None else 0

            # Get starving worker count from metadata
            cursor = conn.execute(
                """
                SELECT value FROM metadata WHERE key = 'starving_workers'
                """
            )
            row = cursor.fetchone()
            starving_workers = int(row[0]) if row else 0

            # Count total splits by counting distinct parent units
            # This includes both splits with progress (parent='completed')
            # and splits without progress (parent='split')
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT parent_id)
                FROM work_units
                WHERE parent_id IS NOT NULL
                """
            )
            total_splits = cursor.fetchone()[0]

            return WorkProgress(
                total=total,
                pending=counts.get('pending', 0),
                processing=counts.get('processing', 0),
                completed=counts.get('completed', 0),
                failed=counts.get('failed', 0),
                split=total_splits,  # Total number of split operations performed
                active_workers=active_workers,
                idle_workers=idle_workers,
                starving_workers=starving_workers,
            )

    def clear_all_work_units(self) -> None:
        """Clear all work units from the tracker."""
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
            conn.execute("DELETE FROM work_units")
            conn.commit()

    def reset_all_processing_units(self) -> int:
        """
        Reset all processing units back to pending.

        Units that were split (have children) are marked as 'split' instead of 'pending'.

        Returns:
            Number of units reset
        """
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
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

    def split_work_unit(self, unit_id: str, max_retries: int = 5) -> tuple[WorkUnit] | tuple[WorkUnit, WorkUnit]:
        """
        Split a work unit, preserving any partial progress.

        If the parent unit has a current_position (partial progress), we save that
        work and create only ONE child unit for the remaining range. Otherwise,
        we split the full range into two child units.

        NOTE: When a unit is split during processing, the parent's partial output
        is preserved as a complete shard. The parent worker should NOT clean up
        its output when detecting a split.

        Args:
            unit_id: ID of unit to split
            max_retries: Maximum number of retry attempts for database locks

        Returns:
            Tuple of child WorkUnit(s) - either (remaining_unit,) or (left_unit, right_unit)

        Raises:
            ValueError: If unit cannot be split
        """
        import time

        for attempt in range(max_retries):
            try:
                return self._split_work_unit_impl(unit_id)
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    continue
                raise

    def _split_work_unit_impl(self, unit_id: str) -> tuple[WorkUnit] | tuple[WorkUnit, WorkUnit]:
        """Implementation of split_work_unit (wrapped with retry logic)."""
        import time

        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
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
            current_position = row['current_position']

            # If unit has made progress, preserve it and split only the remaining range
            if current_position is not None and current_position != start_key:
                # Parent has processed [start_key, current_position] inclusive
                # Child must start AFTER current_position to avoid duplicates
                # Since range_scan uses inclusive lower bound, we need next_key(current_position)

                # Verify we can actually split here
                if current_position == end_key:
                    raise ValueError(
                        f"Unit {unit_id} has already completed its range "
                        f"(current_position == end_key)"
                    )

                # Compute next key after current_position by appending \x00
                # This ensures child starts strictly after parent's last processed key
                # Example: if current_position = b"abc", child starts at b"abc\x00"
                child_start_key = current_position + b'\x00'

                # Verify child range is non-empty
                if end_key is not None and child_start_key >= end_key:
                    raise ValueError(
                        f"Unit {unit_id} cannot be split: "
                        f"next key after current_position would exceed end_key"
                    )

                # Create child unit for remaining work
                from ngram_prep.parallel.partitioning import make_unit_id
                child_id = make_unit_id(child_start_key, end_key)

                # Mark original unit as completed (it processed [start_key, current_position])
                # Only update if still in processing/pending state to avoid race conditions
                cursor = conn.execute(
                    """
                    UPDATE work_units
                    SET status = 'completed', completed_at = ?
                    WHERE unit_id = ? AND status IN ('processing', 'pending')
                    """,
                    (time.time(), unit_id)
                )

                if cursor.rowcount == 0:
                    # Unit was already completed or in another state - can't split
                    raise ValueError(
                        f"Unit {unit_id} status changed during split (may have been completed by worker)"
                    )

                # Insert child unit for remaining range (child_start_key, end_key)
                conn.execute(
                    """
                    INSERT INTO work_units
                        (unit_id, start_key, end_key, status, parent_id)
                    VALUES (?, ?, ?, 'pending', ?)
                    """,
                    (child_id, child_start_key, end_key, unit_id)
                )

                conn.commit()

                # Return single child unit
                return (WorkUnit(child_id, child_start_key, end_key, parent_id=unit_id),)

            # No progress yet - split the full range into two halves
            from ngram_prep.parallel.partitioning import find_midpoint_key
            midpoint = find_midpoint_key(start_key, end_key)

            if midpoint is None:
                start_hex = start_key.hex() if start_key else "None"
                end_hex = end_key.hex() if end_key else "None"
                raise ValueError(
                    f"Unit {unit_id} cannot be split further: "
                    f"no midpoint between {start_hex} and {end_hex}"
                )

            if midpoint == start_key or midpoint == end_key:
                start_hex = start_key.hex() if start_key else "None"
                end_hex = end_key.hex() if end_key else "None"
                mid_hex = midpoint.hex() if midpoint else "None"
                raise ValueError(
                    f"Unit {unit_id} cannot be split further: "
                    f"midpoint {mid_hex} equals boundary (start={start_hex}, end={end_hex})"
                )

            # Create two new units with compact IDs
            from ngram_prep.parallel.partitioning import make_unit_id
            left_id = make_unit_id(start_key, midpoint)
            right_id = make_unit_id(midpoint, end_key)

            # Mark original unit as split
            # Only update if still in processing/pending state to avoid race conditions
            cursor = conn.execute(
                """
                UPDATE work_units
                SET status = 'split'
                WHERE unit_id = ? AND status IN ('processing', 'pending')
                """,
                (unit_id,)
            )

            if cursor.rowcount == 0:
                # Unit was already completed or in another state - can't split
                raise ValueError(
                    f"Unit {unit_id} status changed during split (may have been completed by worker)"
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

        # NOTE: Parent output should be cleaned up only if no progress was saved

        return (
            WorkUnit(left_id, start_key, midpoint, parent_id=unit_id),
            WorkUnit(right_id, midpoint, end_key, parent_id=unit_id)
        )

    def get_any_splittable_unit(self) -> Optional[str]:
        """Get any pending or processing work unit that can be split."""
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
            # Prefer processing units (actively being worked on, likely larger)
            # No preference for parent vs child - split whatever needs splitting
            cursor = conn.execute(
                """
                SELECT unit_id
                FROM work_units
                WHERE status IN ('processing', 'pending')
                ORDER BY
                    CASE status WHEN 'processing' THEN 0 ELSE 1 END,
                    RANDOM()
                    LIMIT 1
                """
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def get_all_splittable_units(self, prefer_processing: bool = True) -> list[str]:
        """Get all processing work units that might be splittable.

        Args:
            prefer_processing: Deprecated parameter kept for compatibility (always uses processing)

        Returns:
            List of unit IDs in priority order (only processing units)
        """
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
            cursor = conn.execute(
                """
                SELECT unit_id
                FROM work_units
                WHERE status = 'processing'
                ORDER BY unit_id
                """
            )
            return [row[0] for row in cursor]

    def get_unit_status(self, unit_id: str) -> Optional[str]:
        """
        Get the current status of a work unit.

        Args:
            unit_id: ID of the work unit

        Returns:
            Status string or None if unit not found
        """
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
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
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
            cursor = conn.execute(
                """
                SELECT unit_id
                FROM work_units
                WHERE status = 'failed'
                """
            )
            return [row[0] for row in cursor]

    def set_starving_workers(self, count: int, max_retries: int = 5) -> None:
        """
        Set the current count of starving workers.

        Args:
            count: Number of workers currently starving (idle for >= starvation_threshold checks)
            max_retries: Maximum number of retry attempts for database locks
        """
        import time

        for attempt in range(max_retries):
            try:
                with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO metadata (key, value)
                        VALUES ('starving_workers', ?)
                        """,
                        (str(count),)
                    )
                    conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.01 * (2 ** attempt))  # Shorter backoff for metadata updates
                    continue
                # If all retries fail, just log and continue - this is non-critical
                if attempt == max_retries - 1:
                    return  # Silently fail - stale count is acceptable

    def get_starving_workers(self) -> int:
        """
        Get the current count of starving workers.

        Returns:
            Number of workers currently starving, or 0 if not set
        """
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
            cursor = conn.execute(
                """
                SELECT value FROM metadata WHERE key = 'starving_workers'
                """
            )
            row = cursor.fetchone()
            return int(row[0]) if row else 0