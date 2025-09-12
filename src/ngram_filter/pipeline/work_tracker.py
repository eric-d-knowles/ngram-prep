# ngram_filter/pipeline/work_tracker.py
"""Work unit tracking system for distributed ngram processing."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from common_db.api import open_db, range_scan


@dataclass
class WorkUnit:
    """Represents a unit of work for processing a key range."""

    unit_id: str
    start_key: Optional[bytes]
    end_key: Optional[bytes]
    status: str = "pending"  # pending, processing, completed, failed


@dataclass
class WorkProgress:
    """Progress statistics for work units."""

    completed: int
    processing: int
    pending: int
    failed: int = 0

    @property
    def total(self) -> int:
        """Total number of work units."""
        return self.completed + self.processing + self.pending + self.failed

    @property
    def completion_rate(self) -> float:
        """Percentage of work completed (0.0 to 1.0)."""
        if self.total == 0:
            return 0.0
        return self.completed / self.total


class WorkTracker:
    """Persistent work unit tracking using SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Create the work tracking database and tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS work_units
                         (
                             unit_id
                             TEXT
                             PRIMARY
                             KEY,
                             start_key
                             BLOB,
                             end_key
                             BLOB,
                             status
                             TEXT
                             DEFAULT
                             'pending',
                             worker_id
                             TEXT,
                             start_time
                             REAL,
                             end_time
                             REAL,
                             retry_count
                             INTEGER
                             DEFAULT
                             0
                         )
                         """)

            # Create indexes for efficient querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON work_units(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_worker ON work_units(worker_id)")

    def add_work_units(self, work_units: list[WorkUnit]) -> None:
        """
        Add multiple work units to the tracker.

        Args:
            work_units: List of WorkUnit objects to add
        """
        if not work_units:
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT
                OR IGNORE INTO work_units (unit_id, start_key, end_key, status)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (unit.unit_id, unit.start_key, unit.end_key, unit.status)
                    for unit in work_units
                ]
            )

    def claim_work_unit(self, worker_id: str) -> Optional[WorkUnit]:
        """
        Atomically claim the next available work unit.

        Args:
            worker_id: Identifier for the worker claiming the unit

        Returns:
            The claimed WorkUnit or None if no work is available
        """
        with sqlite3.connect(self.db_path) as conn:
            # Use RETURNING clause for atomic claim operation
            cursor = conn.execute(
                """
                UPDATE work_units
                SET status     = 'processing',
                    worker_id  = ?,
                    start_time = julianday('now')
                WHERE unit_id = (SELECT unit_id
                                 FROM work_units
                                 WHERE status = 'pending'
                                 ORDER BY unit_id
                    LIMIT 1
                    )
                    RETURNING unit_id
                    , start_key
                    , end_key
                """,
                (worker_id,)
            )

            row = cursor.fetchone()
            if row:
                return WorkUnit(
                    unit_id=row[0],
                    start_key=row[1],
                    end_key=row[2],
                    status="processing"
                )
            return None

    def complete_work_unit(self, unit_id: str) -> None:
        """
        Mark a work unit as successfully completed.

        Args:
            unit_id: ID of the work unit to complete
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE work_units
                SET status   = 'completed',
                    end_time = julianday('now')
                WHERE unit_id = ?
                """,
                (unit_id,)
            )

    def fail_work_unit(self, unit_id: str, max_retries: int = 3) -> None:
        """
        Mark a work unit as failed and reset for retry if under retry limit.

        Args:
            unit_id: ID of the work unit that failed
            max_retries: Maximum number of retry attempts
        """
        with sqlite3.connect(self.db_path) as conn:
            # Increment retry count and check if we should retry
            cursor = conn.execute(
                """
                SELECT retry_count
                FROM work_units
                WHERE unit_id = ?
                """,
                (unit_id,)
            )

            row = cursor.fetchone()
            if not row:
                return

            retry_count = row[0] + 1

            if retry_count <= max_retries:
                # Reset for retry
                conn.execute(
                    """
                    UPDATE work_units
                    SET status      = 'pending',
                        worker_id   = NULL,
                        start_time  = NULL,
                        retry_count = ?
                    WHERE unit_id = ?
                    """,
                    (retry_count, unit_id)
                )
            else:
                # Mark as permanently failed
                conn.execute(
                    """
                    UPDATE work_units
                    SET status      = 'failed',
                        end_time    = julianday('now'),
                        retry_count = ?
                    WHERE unit_id = ?
                    """,
                    (retry_count, unit_id)
                )

    def get_progress(self) -> WorkProgress:
        """
        Get current progress statistics.

        Returns:
            WorkProgress object with current counts
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT status, COUNT(*)
                FROM work_units
                GROUP BY status
                """
            )

            counts = {
                "completed": 0,
                "processing": 0,
                "pending": 0,
                "failed": 0
            }

            for status, count in cursor.fetchall():
                if status in counts:
                    counts[status] = count

            return WorkProgress(
                completed=counts["completed"],
                processing=counts["processing"],
                pending=counts["pending"],
                failed=counts["failed"]
            )

    def get_stuck_work_units(self, timeout_hours: float = 1.0) -> list[str]:
        """
        Find work units that have been processing for too long.

        Args:
            timeout_hours: Hours after which a processing unit is considered stuck

        Returns:
            List of unit IDs that appear to be stuck
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT unit_id
                FROM work_units
                WHERE status = 'processing'
                  AND start_time < julianday('now') - ?
                """,
                (timeout_hours / 24,)  # Convert hours to days for Julian day arithmetic
            )

            return [row[0] for row in cursor.fetchall()]

    def reset_stuck_work_units(self, timeout_hours: float = 1.0) -> int:
        """
        Reset stuck work units back to pending status.

        Args:
            timeout_hours: Hours after which a processing unit is considered stuck

        Returns:
            Number of work units that were reset
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE work_units
                SET status     = 'pending',
                    worker_id  = NULL,
                    start_time = NULL
                WHERE status = 'processing'
                  AND start_time < julianday('now') - ?
                """,
                (timeout_hours / 24,)
            )

            return cursor.rowcount

    def clear_all_work_units(self) -> None:
        """Remove all work units from the tracker."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM work_units")


def create_work_units(src_db_path: Path, num_units: int = 128) -> list[WorkUnit]:
    """
    Create work units using ASCII printable character range.

    Args:
        src_db_path: Path to source database (for validation)
        num_units: Number of work units to create

    Returns:
        List of WorkUnit objects covering the key space
    """
    print(f"  Creating {num_units} work units using ASCII range...")

    # Define the printable ASCII range that contains actual data
    first_byte = 0x21  # "!" character
    last_byte = 0x7E  # "~" character
    byte_range = last_byte - first_byte

    work_units = []

    for i in range(num_units):
        # Determine start key
        if i == 0:
            start_key = None  # Start from beginning
        else:
            byte_value = first_byte + (i * byte_range) // num_units
            start_key = bytes([byte_value])

        # Determine end key
        if i == num_units - 1:
            end_key = None  # Go to end
        else:
            byte_value = first_byte + ((i + 1) * byte_range) // num_units
            end_key = bytes([byte_value])

        work_units.append(WorkUnit(
            unit_id=f"unit_{i:04d}",
            start_key=start_key,
            end_key=end_key
        ))

    print(f"  Created {len(work_units)} work units covering range 0x{first_byte:02x}-0x{last_byte:02x}")
    return work_units


def validate_work_units(src_db_path: Path, work_units: list[WorkUnit]) -> bool:
    """
    Validate that work units can access their assigned key ranges.

    Args:
        src_db_path: Path to the source database
        work_units: List of work units to validate

    Returns:
        True if validation passes, False otherwise
    """
    print(f"  Validating {len(work_units)} work units...")

    try:
        with open_db(src_db_path, mode="ro") as db:
            total_keys_found = 0
            validation_sample_size = min(10, len(work_units))

            for i in range(validation_sample_size):
                unit = work_units[i]
                start_repr = _format_key_for_display(unit.start_key)

                #print(f"  Validating work unit {i}: {start_repr}...")

                keys_in_range = _count_keys_in_range(db, unit)
                if keys_in_range < 0:  # Error occurred
                    return False

                #print(f"    Found {keys_in_range} keys in range")
                total_keys_found += keys_in_range

            print(f"  Validated {len(work_units)} work units: {total_keys_found} keys over {validation_sample_size} sample units")
            return total_keys_found > 0

    except Exception as e:
        print(f"Validation failed: {e}")
        return False


def _format_key_for_display(key: Optional[bytes]) -> str:
    """Format a key for human-readable display."""
    if key is None:
        return "beginning"
    return key.hex()[:8] + ("..." if len(key) > 4 else "")


def _count_keys_in_range(db, work_unit: WorkUnit, max_sample: int = 100) -> int:
    """
    Count keys in a work unit's range, up to max_sample.

    Returns:
        Number of keys found, or -1 if an error occurred
    """
    try:
        start_key = work_unit.start_key if work_unit.start_key is not None else b""
        count = 0

        for _ in range_scan(db, start_key, work_unit.end_key):
            count += 1
            if count >= max_sample:
                break

        return count

    except Exception as e:
        print(f"    Error scanning range: {e}")
        return -1