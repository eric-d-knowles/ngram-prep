"""Work unit splitting logic for dynamic load balancing."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

from .types import WorkUnit
from .partitioning import find_midpoint_key, make_unit_id

__all__ = ["WorkSplitter"]


class WorkSplitter:
    """Handles splitting of work units for load balancing."""

    def __init__(self, db_path: Path):
        """
        Initialize the work splitter.

        Args:
            db_path: Path to SQLite tracking database
        """
        self.db_path = Path(db_path)

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
        import random

        for attempt in range(max_retries):
            try:
                return self._split_current_unit_impl(unit_id)
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    base_delay = 0.1 * (2 ** attempt)
                    jitter = random.uniform(0, base_delay * 0.5)
                    time.sleep(base_delay + jitter)
                    continue
                raise

        return None

    def _split_current_unit_impl(self, unit_id: str) -> Optional[WorkUnit]:
        """Implementation of split_current_unit (wrapped with retry logic)."""
        with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
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
            split_point = find_midpoint_key(start_key, end_key)
            if split_point is None:
                return None  # Can't find midpoint

            # Parent gets [start_key, split_point), child gets [split_point, end_key)
            parent_end = split_point
            child_start = split_point

            # Verify child range is non-empty
            if end_key is not None and child_start >= end_key:
                return None  # No room for child

            # Create child unit for remaining work
            child_id = make_unit_id(child_start, end_key)

            # Shrink parent's range
            cursor = conn.execute(
                """
                UPDATE work_units
                SET end_key = ?
                WHERE unit_id = ?
                  AND status = 'processing'
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
        import random

        for attempt in range(max_retries):
            try:
                return self._split_work_unit_impl(unit_id)
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    base_delay = 0.1 * (2 ** attempt)
                    jitter = random.uniform(0, base_delay * 0.5)
                    time.sleep(base_delay + jitter)
                    continue
                raise

    def _split_work_unit_impl(self, unit_id: str) -> WorkUnit:
        """Implementation of split_work_unit (wrapped with retry logic)."""
        with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
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

            # Verify we can actually split here
            if current_position == end_key:
                raise ValueError(
                    f"Unit {unit_id} has already completed its range "
                    f"(current_position == end_key)"
                )

            # Compute next key after current_position by appending \x00
            # This ensures child starts strictly after parent's last processed key
            child_start_key = current_position + b'\x00'

            # Verify child range is non-empty
            if end_key is not None and child_start_key >= end_key:
                raise ValueError(
                    f"Unit {unit_id} cannot be split: "
                    f"next key after current_position would exceed end_key"
                )

            # Create child unit for remaining work
            child_id = make_unit_id(child_start_key, end_key)

            # Shrink parent's range to current_position and mark as completed
            # Parent has processed [start_key, current_position] and is done
            # Worker will detect the split and exit its scan loop
            cursor = conn.execute(
                """
                UPDATE work_units
                SET end_key = ?,
                    status = 'completed'
                WHERE unit_id = ?
                  AND status = 'processing'
                """,
                (current_position, unit_id)
            )

            if cursor.rowcount == 0:
                # Unit was already completed or in another state - can't split
                raise ValueError(
                    f"Unit {unit_id} status changed during split (may have been completed by worker)"
                )

            # Insert child unit for remaining range
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
