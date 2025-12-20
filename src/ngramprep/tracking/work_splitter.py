"""Work unit splitting logic for dynamic load balancing."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

from .types import WorkUnit
from .partitioning import make_unit_id

__all__ = ["WorkSplitter"]

# Metadata prefix for filtering out system keys
METADATA_PREFIX = b"__"


class WorkSplitter:
    """Handles splitting of work units for load balancing."""

    def __init__(self, db_path: Path):
        """
        Initialize the work splitter.

        Args:
            db_path: Path to SQLite tracking database
        """
        self.db_path = Path(db_path)

    def split_work_unit(
        self,
        unit_id: str,
        max_retries: int = 5
    ) -> tuple[WorkUnit, bytes]:
        """
        Split a work unit's remaining range in half geometrically.

        Calculates the geometric midpoint between current_position and end_key.
        Creates a child unit for [midpoint, end_key) and shrinks parent to
        [start_key, midpoint). Parent worker continues processing with the
        shrunk range.

        Requires the unit to have made progress (current_position set).

        Args:
            unit_id: ID of unit to split
            max_retries: Maximum number of retry attempts for database locks

        Returns:
            Tuple of (child WorkUnit, new_parent_end_key)
            - child: WorkUnit for [midpoint, end_key)
            - new_parent_end_key: Midpoint where parent should stop

        Raises:
            ValueError: If unit cannot be split (no progress, invalid state, range too small, etc.)
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

    def _split_work_unit_impl(
        self,
        unit_id: str
    ) -> tuple[WorkUnit, bytes]:
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

            # For split-at-claim (no progress yet), split between start and end
            # For mid-processing splits, split between checkpoint and end
            split_start = current_position if current_position is not None else start_key

            # Verify we can actually split here
            if split_start == end_key:
                raise ValueError(
                    f"Unit {unit_id} has already completed its range "
                    f"(split_start == end_key)"
                )

            # Calculate geometric midpoint between split_start and end_key
            midpoint = self._calculate_midpoint(split_start, end_key)

            # Verify midpoint is valid (not at boundaries)
            if midpoint <= split_start or (end_key is not None and midpoint >= end_key):
                raise ValueError(
                    f"Unit {unit_id} cannot be split: "
                    f"remaining range too small for geometric split"
                )

            # Create child unit for [midpoint, end_key)
            child_start_key = midpoint
            child_id = make_unit_id(child_start_key, end_key)

            # Shrink parent's end_key to midpoint (parent continues processing)
            # Parent will process [start_key, midpoint), NOT marked as completed
            cursor = conn.execute(
                """
                UPDATE work_units
                SET end_key = ?
                WHERE unit_id = ?
                  AND status = 'processing'
                """,
                (midpoint, unit_id)
            )

            if cursor.rowcount == 0:
                # Unit was already completed or in another state - can't split
                raise ValueError(
                    f"Unit {unit_id} status changed during split (may have been completed by worker)"
                )

            # Insert child unit for [midpoint, end_key)
            conn.execute(
                """
                INSERT INTO work_units
                (unit_id, start_key, end_key, status, parent_id)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (child_id, child_start_key, end_key, unit_id)
            )

            conn.commit()

            # Return child unit and new parent end_key (midpoint)
            return WorkUnit(child_id, child_start_key, end_key, parent_id=unit_id), midpoint

    def _calculate_midpoint(self, start: bytes, end: Optional[bytes]) -> bytes:
        """
        Calculate geometric midpoint between two byte strings.

        Args:
            start: Start key (inclusive)
            end: End key (exclusive), or None for unbounded

        Returns:
            Midpoint byte string

        Raises:
            ValueError: If range is too small to split
        """
        if end is None:
            # No end key - append byte to start to create midpoint
            return start + b'\x80'

        # Ensure end > start
        if end <= start:
            raise ValueError(f"Range too small: end <= start")

        # Try split strategies from largest to smallest
        # This ensures we create substantial work units for idle workers

        # Strategy 1: Geometric midpoint (largest split - try this first!)
        # Pad to same length for arithmetic
        max_len = max(len(start), len(end))
        start_padded = start.ljust(max_len, b'\x00')
        end_padded = end.ljust(max_len, b'\x00')

        # Convert to integers
        start_int = int.from_bytes(start_padded, 'big')
        end_int = int.from_bytes(end_padded, 'big')

        # Calculate midpoint
        mid_int = (start_int + end_int) // 2

        # Check if geometric midpoint is valid
        if mid_int > start_int and mid_int < end_int:
            # Convert back to bytes
            mid_bytes = mid_int.to_bytes(max_len, 'big')

            # Trim trailing zeros conservatively
            while len(mid_bytes) > 1 and mid_bytes[-1] == 0:
                trimmed = mid_bytes[:-1]
                if trimmed <= start or trimmed >= end:
                    break
                mid_bytes = trimmed

            return mid_bytes

        # Strategy 2: Append a byte to start (fallback for small ranges)
        candidate = start + b'\x01'
        if candidate < end:
            return candidate

        # Strategy 3: Byte-level increment (smallest possible split, last resort)
        if len(start) > 0 and start[-1] < 255:
            candidate = start[:-1] + bytes([start[-1] + 1])
            if candidate < end:
                return candidate

        # If nothing worked, range is too small
        raise ValueError(f"Range too small: no valid midpoint exists")
