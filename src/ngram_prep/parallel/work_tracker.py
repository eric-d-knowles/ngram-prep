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

    def checkpoint_and_check_split(self, unit_id: str, position: bytes, max_retries: int = 5) -> bool:
        """
        Atomically update checkpoint position and check if unit was split.

        This is the ONLY safe way for a worker to checkpoint without risk of double-counting.
        The operation is atomic within a single transaction, ensuring that if the split monitor
        marks the unit as 'split', this method will detect it before the worker commits counts.

        Args:
            unit_id: ID of work unit
            position: Current scan position (key) - worker has processed up to and including this key
            max_retries: Maximum number of retry attempts for database locks

        Returns:
            True if unit was split (worker should exit without committing counts),
            False if unit is still processing (worker should commit counts)
        """
        import time

        for attempt in range(max_retries):
            try:
                with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                    conn.row_factory = sqlite3.Row

                    # Update checkpoint and retrieve status in single transaction
                    conn.execute(
                        """
                        UPDATE work_units
                        SET current_position = ?
                        WHERE unit_id = ?
                        """,
                        (position, unit_id)
                    )

                    # Get status in same transaction
                    cursor = conn.execute(
                        """
                        SELECT status
                        FROM work_units
                        WHERE unit_id = ?
                        """,
                        (unit_id,)
                    )

                    row = cursor.fetchone()
                    conn.commit()

                    if not row:
                        raise ValueError(f"Unit {unit_id} not found")

                    return row['status'] == 'split'

            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                    continue
                raise

        raise RuntimeError(f"Failed to checkpoint after {max_retries} retries")

    def complete_work_unit(self, unit_id: str, max_retries: int = 5) -> None:
        """
        Mark a work unit as completed (finished scanning its key range).

        Transitions from 'processing' or 'split' to 'completed'.

        Args:
            unit_id: ID of completed work unit
            max_retries: Maximum number of retry attempts for database locks

        Raises:
            ValueError: If unit is not in valid state for completion
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
                        WHERE unit_id = ? AND status IN ('processing', 'split')
                        """,
                        (time.time(), unit_id)
                    )

                    if cursor.rowcount == 0:
                        # Check if unit is in another state
                        cursor = conn.execute(
                            "SELECT status FROM work_units WHERE unit_id = ?",
                            (unit_id,)
                        )
                        row = cursor.fetchone()
                        if row:
                            current_status = row[0]
                            if current_status == 'completed':
                                # Already completed - this is OK (idempotent)
                                conn.commit()
                                return
                            else:
                                raise ValueError(
                                    f"Cannot complete unit {unit_id}: "
                                    f"status is '{current_status}' (expected 'processing' or 'split')"
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

    def ingest_work_unit(self, unit_id: str, max_retries: int = 5) -> None:
        """
        Mark a work unit as ingested (successfully added to destination DB).

        Transitions directly from 'processing'/'split' to 'ingested'.

        Args:
            unit_id: ID of work unit to mark as ingested
            max_retries: Maximum number of retry attempts for database locks
        """
        import time

        for attempt in range(max_retries):
            try:
                with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                    cursor = conn.execute(
                        """
                        UPDATE work_units
                        SET status = 'ingested',
                            completed_at = ?
                        WHERE unit_id = ? AND status IN ('processing', 'split')
                        """,
                        (time.time(), unit_id)
                    )

                    if cursor.rowcount == 0:
                        # Check if already ingested (idempotent)
                        cursor = conn.execute(
                            "SELECT status FROM work_units WHERE unit_id = ?",
                            (unit_id,)
                        )
                        row = cursor.fetchone()
                        if row and row[0] == 'ingested':
                            conn.commit()
                            return
                        # Otherwise fall through to commit (unit may have been in unexpected state)

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
            # - Parent units with status='completed'/'ingested'/'split' (split units still being processed)
            cursor = conn.execute(
                """
                SELECT status, COUNT(*) as count
                FROM work_units
                WHERE (
                    -- Include leaf units (not yet split into children)
                    unit_id NOT IN (
                        SELECT DISTINCT parent_id
                        FROM work_units
                        WHERE parent_id IS NOT NULL
                    )
                    OR
                    -- Include parent units that are still processing or finished
                    -- (status in 'completed', 'ingested', 'split')
                    (status IN ('completed', 'ingested', 'split') AND unit_id IN (
                        SELECT DISTINCT parent_id
                        FROM work_units
                        WHERE parent_id IS NOT NULL
                    ))
                )
                GROUP BY status
                """
            )

            counts = {row[0]: row[1] for row in cursor}

            # Map 'split' status to 'processing' for display purposes
            # These units are still being actively processed by workers
            if 'split' in counts:
                counts['processing'] = counts.get('processing', 0) + counts['split']
                del counts['split']
            
            # Count units in 'split' status separately (they're awaiting worker finalization)
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM work_units WHERE status = 'split'
                """
            )
            splitting_count = cursor.fetchone()[0]
            
            total = sum(counts.values())

            # Count distinct workers actively processing units
            # Include both 'processing' and 'split' status (split units are still being processed)
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT claimed_by)
                FROM work_units
                WHERE status IN ('processing', 'split') AND claimed_by IS NOT NULL
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
                saved=0,  # No longer used: inline ingestion
                ingested=counts.get('ingested', 0),
                split=total_splits,  # Total number of split operations performed
                splitting=splitting_count,  # Number of units currently in 'split' status
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
        Reset all processing units back to pending on restart.

        Units that have children (were split) remain 'completed'.
        Units without children are reset to 'pending'.

        Returns:
            Number of units reset
        """
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
            # First, mark any processing units that have children as completed
            # (these were split but the worker didn't finish marking them ingested before crash)
            conn.execute(
                """
                UPDATE work_units
                SET status = 'completed'
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

    def split_current_unit(self, unit_id: str, max_retries: int = 5) -> Optional[WorkUnit]:
        """
        Split a processing work unit at its midpoint, creating a child for the remainder.

        This is the worker-driven split approach: when a worker claims a unit and detects
        idle workers, it immediately splits the unit at its midpoint before processing begins.

        Split behavior:
        - Parent gets [start_key, midpoint)
        - Child gets [midpoint, end_key) with status='pending'
        - Both parent and child process their respective ranges from scratch
        - No data duplication since ranges are non-overlapping

        The parent continues to process its (now smaller) range normally.
        The child unit is added to the pending queue for other workers to claim.

        Note: This is called immediately after a unit is claimed, before any processing
        begins, so the unit never has progress (current_position is always None).

        Args:
            unit_id: ID of unit to split
            max_retries: Maximum number of retry attempts for database locks

        Returns:
            Child WorkUnit for the remaining range, or None if unit cannot be split

        Raises:
            ValueError: If unit is not in valid state for splitting
        """
        import time

        for attempt in range(max_retries):
            try:
                return self._split_current_unit_impl(unit_id)
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                raise

        return None

    def _split_current_unit_impl(self, unit_id: str) -> Optional[WorkUnit]:
        """Implementation of split_current_unit (wrapped with retry logic)."""
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

            if row['status'] != 'processing':
                raise ValueError(f"Unit {unit_id} status is {row['status']}, can only split processing units")

            start_key = row['start_key']
            end_key = row['end_key']

            # Split at midpoint between start_key and end_key
            # Note: This is called immediately after unit is claimed, so no progress yet
            from ngram_prep.parallel.partitioning import find_midpoint_key
            split_point = find_midpoint_key(start_key, end_key)
            if split_point is None:
                return None  # Can't find midpoint (e.g., both keys are None)

            # Parent gets [start_key, split_point)
            # Child gets [split_point, end_key)
            parent_end = split_point
            child_start = split_point

            # Verify child range is non-empty
            if end_key is not None and child_start >= end_key:
                return None  # No room for child

            # Create child unit for remaining work
            from ngram_prep.parallel.partitioning import make_unit_id
            child_id = make_unit_id(child_start, end_key)

            # Shrink parent's range
            cursor = conn.execute(
                """
                UPDATE work_units
                SET end_key = ?
                WHERE unit_id = ? AND status = 'processing'
                """,
                (parent_end, unit_id)
            )

            if cursor.rowcount == 0:
                return None  # Unit changed status, can't split

            # Insert child unit for remaining range
            conn.execute(
                """
                INSERT INTO work_units
                    (unit_id, start_key, end_key, status, parent_id)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (child_id, child_start, end_key, unit_id)
            )

            conn.commit()

            return WorkUnit(child_id, child_start, end_key, parent_id=unit_id)

    def split_work_unit(self, unit_id: str, max_retries: int = 5) -> WorkUnit:
        """
        Split a work unit with progress, preserving partial work.

        Requires the unit to have made progress (current_position set).
        Creates ONE child unit for the remaining range after current_position.
        Parent unit is marked as 'completed' immediately since all data up to
        current_position is already flushed to disk.

        The worker will detect the status change, exit its scan loop early, and
        finalize the shard (mark as 'ingested').

        NOTE: When a unit is split during processing, the parent's partial output
        is preserved as a complete shard. The parent worker should NOT clean up
        its output when detecting the split.

        Args:
            unit_id: ID of unit to split
            max_retries: Maximum number of retry attempts for database locks

        Returns:
            Child WorkUnit for the remaining range

        Raises:
            ValueError: If unit cannot be split (no progress, invalid state, etc.)
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

    def _split_work_unit_impl(self, unit_id: str) -> WorkUnit:
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

            if row['status'] != 'processing':
                raise ValueError(f"Unit {unit_id} status is {row['status']}, can only split processing units")

            start_key = row['start_key']
            end_key = row['end_key']
            current_position = row['current_position']

            # Require progress to split
            if current_position is None or current_position == start_key:
                raise ValueError(
                    f"Unit {unit_id} has no progress, cannot split "
                    f"(current_position is None or equals start_key)"
                )

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

            # Mark original unit as 'split' (split decision made, awaiting worker finalization)
            # Worker will detect 'split' status after its next flush and transition to 'completed'
            # Only update if still in processing state to avoid race conditions
            cursor = conn.execute(
                """
                UPDATE work_units
                SET status = 'split'
                WHERE unit_id = ? AND status = 'processing'
                """,
                (unit_id,)
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

            return WorkUnit(child_id, child_start_key, end_key, parent_id=unit_id)

    def get_any_splittable_unit(self) -> Optional[str]:
        """Get any processing work unit that can be split."""
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
            # Only processing units can be split (they have active workers checkpointing progress)
            cursor = conn.execute(
                """
                SELECT unit_id
                FROM work_units
                WHERE status = 'processing'
                ORDER BY RANDOM()
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def get_all_splittable_units(self, prefer_processing: bool = True) -> list[str]:
        """Get all processing work units that might be splittable.

        Only processing units are eligible for splitting since they have active workers
        checkpointing progress.

        Args:
            prefer_processing: Deprecated parameter kept for compatibility (ignored)

        Returns:
            List of processing unit IDs
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

    def get_work_unit(self, unit_id: str) -> Optional[WorkUnit]:
        """
        Get a work unit by ID.

        Args:
            unit_id: ID of the work unit

        Returns:
            WorkUnit or None if not found
        """
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
            cursor = conn.execute(
                "SELECT unit_id, start_key, end_key, parent_id, current_position FROM work_units WHERE unit_id = ?",
                (unit_id,)
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
            return None

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

    def get_completed_unit_ids(self) -> list[str]:
        """
        Get list of completed unit IDs (both leaf units and split parents with progress).

        Returns:
            List of unit IDs with status 'completed'
        """
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
            cursor = conn.execute(
                """
                SELECT unit_id
                FROM work_units
                WHERE status = 'completed'
                ORDER BY completed_at
                """
            )
            return [row[0] for row in cursor]

    def get_ingested_unit_ids(self) -> list[str]:
        """
        Get list of ingested unit IDs (successfully added to destination DB).

        Returns:
            List of unit IDs with status 'ingested'
        """
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
            cursor = conn.execute(
                """
                SELECT unit_id
                FROM work_units
                WHERE status = 'ingested'
                ORDER BY unit_id
                """
            )
            return [row[0] for row in cursor]