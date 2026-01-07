"""CRUD operations for work units in the tracking database."""

from __future__ import annotations

import random
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

from .types import WorkUnit

__all__ = ["WorkUnitStore"]


class WorkUnitStore:
    """Handles database operations for work units."""

    def __init__(self, db_path: Path, claim_order: str = "sequential"):
        """
        Initialize the work unit store.

        Args:
            db_path: Path to SQLite tracking database
            claim_order: Order for claiming work units - "sequential" or "random"
        """
        self.db_path = Path(db_path)
        self.claim_order = claim_order

    def _connect(self, timeout: float = 60.0) -> sqlite3.Connection:
        """
        Create a database connection with optimal settings for concurrency.

        Args:
            timeout: Database lock timeout in seconds

        Returns:
            SQLite connection with WAL mode enabled
        """
        conn = sqlite3.connect(str(self.db_path), timeout=timeout)
        # Enable WAL mode for better concurrency on each connection
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def add_work_units(self, work_units: List[WorkUnit]) -> None:
        """
        Add work units to the database.

        Args:
            work_units: List of work units to add
        """
        with self._connect(timeout=30.0) as conn:
            for unit in work_units:
                parent_id = getattr(unit, 'parent_id', None)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO work_units
                    (unit_id, start_key, end_key, status, parent_id)
                    VALUES (?, ?, ?, 'pending', ?)
                    """,
                    (unit.unit_id, unit.start_key, unit.end_key, parent_id)
                )
            conn.commit()

    def claim_work_unit(self, worker_id: str, max_retries: int = 15) -> Optional[WorkUnit]:
        """
        Atomically claim the next available work unit.
        
        Now uses fail-fast approach: if DB is locked, return None immediately
        and let worker-level backoff handle retries.

        Args:
            worker_id: Identifier for the worker claiming the unit
            max_retries: Ignored - kept for API compatibility

        Returns:
            The claimed WorkUnit or None if no work is available or DB is locked
        """
        import random

        # Determine ORDER BY clause based on claim_order
        order_clause = "RANDOM()" if self.claim_order == "random" else "unit_id"

        try:
            # Use very short timeout - fail immediately if locked
            with self._connect(timeout=0.1) as conn:
                # Use immediate transaction to claim lock or fail instantly
                try:
                    conn.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError:
                    # DB is locked - fail fast, let worker retry
                    return None
                
                # First check if there's any pending work at all
                check_cursor = conn.execute(
                    "SELECT COUNT(*) FROM work_units WHERE status = 'pending'"
                )
                pending_count = check_cursor.fetchone()[0]
                
                if pending_count == 0:
                    # No pending work - exit immediately
                    conn.rollback()
                    return None
                
                cursor = conn.execute(
                    f"""
                    UPDATE work_units
                    SET status = 'processing',
                        claimed_by = ?,
                        claimed_at = ?
                    WHERE unit_id = (
                        SELECT unit_id
                        FROM work_units
                        WHERE status = 'pending'
                        ORDER BY {order_clause}
                        LIMIT 1
                    )
                    RETURNING unit_id, start_key, end_key, parent_id, current_position
                    """,
                    (worker_id, time.time())
                )

                row = cursor.fetchone()
                if row:
                    # Success - commit and return
                    conn.commit()
                    return WorkUnit(
                        unit_id=row[0],
                        start_key=row[1],
                        end_key=row[2],
                        parent_id=row[3],
                        current_position=row[4],
                    )
                # No work claimed (race condition)
                conn.rollback()
                return None

        except sqlite3.OperationalError:
            # Any database error - fail fast
            return None

    def get_work_unit(self, unit_id: str) -> Optional[WorkUnit]:
        """
        Get a work unit by ID.

        Args:
            unit_id: ID of the work unit

        Returns:
            WorkUnit or None if not found
        """
        with self._connect(timeout=30.0) as conn:
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

    def checkpoint_position(self, unit_id: str, position: bytes, max_retries: int = 5) -> None:
        """
        Update the current scan position for a work unit.

        Args:
            unit_id: ID of work unit
            position: Current scan position (key)
            max_retries: Maximum number of retry attempts for database locks
        """
        import random

        for attempt in range(max_retries):
            try:
                # Use immediate transaction mode to fail fast on contention
                # instead of waiting for long timeout periods
                conn = self._connect(timeout=10.0)
                conn.isolation_level = 'IMMEDIATE'
                try:
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
                finally:
                    conn.close()
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    # Exponential backoff with jitter to avoid thundering herd
                    base_delay = 0.1 * (2 ** attempt)
                    jitter = random.uniform(0, base_delay * 0.5)
                    time.sleep(base_delay + jitter)
                    continue
                # Checkpoint failures are not fatal - just log and continue
                # The worker can resume from an earlier checkpoint or start of unit
                import sys
                print(f"Warning: Failed to checkpoint position for {unit_id} after {max_retries} attempts",
                      file=sys.stderr)
                return  # Don't raise, just return

    def fail_work_unit(self, unit_id: str, max_retries: int = 5) -> None:
        """
        Mark a work unit as failed.

        Args:
            unit_id: ID of failed work unit
            max_retries: Maximum number of retry attempts for database locks
        """
        import random

        for attempt in range(max_retries):
            try:
                with self._connect(timeout=60.0) as conn:
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
                    base_delay = 0.1 * (2 ** attempt)
                    jitter = random.uniform(0, base_delay * 0.5)
                    time.sleep(base_delay + jitter)
                    continue
                import sys
                # If we can't mark a unit as failed, raise exception - unit would be stuck forever
                raise RuntimeError(
                    f"Failed to mark unit {unit_id} as failed after {max_retries} attempts: {e}"
                ) from e

    def complete_work_unit(self, unit_id: str, max_retries: int = 20) -> None:
        """
        Mark a work unit as completed (finished filtering, ready for ingest).

        Transitions from 'processing' to 'completed'.

        Args:
            unit_id: ID of work unit to mark as completed
            max_retries: Maximum number of retry attempts for database locks
        """
        for attempt in range(max_retries):
            try:
                with sqlite3.connect(str(self.db_path), timeout=60.0) as conn:
                    cursor = conn.execute(
                        """
                        UPDATE work_units
                        SET status = 'completed',
                            completed_at = ?
                        WHERE unit_id = ?
                          AND status = 'processing'
                        """,
                        (time.time(), unit_id)
                    )

                    if cursor.rowcount == 0:
                        # Check if already completed (idempotent)
                        cursor = conn.execute(
                            "SELECT status FROM work_units WHERE unit_id = ?",
                            (unit_id,)
                        )
                        row = cursor.fetchone()
                        if row and row[0] in ('completed', 'ingested'):
                            conn.commit()
                            return

                    conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    # Longer backoff with more jitter for high contention scenarios
                    base_delay = 0.2 * (1.5 ** attempt)
                    jitter = random.uniform(0, base_delay * 0.8)
                    time.sleep(base_delay + jitter)
                    continue
                # If we exhausted retries, raise the exception instead of silently failing
                raise RuntimeError(
                    f"Failed to mark unit {unit_id} as completed after {max_retries} attempts: {e}"
                ) from e

    def get_next_completed_unit(self, max_retries: int = 5) -> Optional[str]:
        """
        Get next completed-but-not-ingested unit ID for ingestion.

        WARNING: This is NOT atomic - use claim_completed_unit_for_ingest() instead
        to avoid race conditions with multiple readers.

        Returns:
            Unit ID of next completed unit, or None if none available
        """
        for attempt in range(max_retries):
            try:
                with self._connect(timeout=30.0) as conn:
                    cursor = conn.execute(
                        """
                        SELECT unit_id
                        FROM work_units
                        WHERE status = 'completed'
                        ORDER BY unit_id
                        LIMIT 1
                        """
                    )
                    row = cursor.fetchone()
                    return row[0] if row else None
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.01 * (2 ** attempt))
                    continue
                raise
        return None

    def claim_completed_unit_for_ingest(self, max_retries: int = 5) -> Optional[str]:
        """
        Atomically claim next completed unit by transitioning 'completed' → 'ingesting'.

        This is atomic - only ONE reader will successfully claim each unit,
        preventing duplicate ingestion by concurrent readers.

        The unit will be in 'ingesting' state until the writer confirms successful
        ingestion by calling ingest_work_unit(), which transitions to 'ingested'.

        Returns:
            Unit ID that was claimed, or None if no completed units available
        """
        for attempt in range(max_retries):
            try:
                with self._connect(timeout=30.0) as conn:
                    # First, find the next completed unit
                    cursor = conn.execute(
                        """
                        SELECT unit_id
                        FROM work_units
                        WHERE status = 'completed'
                        ORDER BY unit_id
                        LIMIT 1
                        """
                    )
                    row = cursor.fetchone()

                    if row is None:
                        return None  # No completed units available

                    unit_id = row[0]

                    # Atomically claim it: transition to 'ingesting' only if still 'completed'
                    cursor = conn.execute(
                        """
                        UPDATE work_units
                        SET status = 'ingesting'
                        WHERE unit_id = ?
                          AND status = 'completed'
                        """,
                        (unit_id,)
                    )

                    if cursor.rowcount == 1:
                        # Successfully claimed!
                        conn.commit()
                        return unit_id
                    else:
                        # Another reader claimed it first - try again
                        conn.rollback()
                        continue

            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.01 * (2 ** attempt))
                    continue
                raise
        return None

    def ingest_work_unit(self, unit_id: str, max_retries: int = 5) -> None:
        """
        Mark a work unit as ingested (successfully added to destination DB).

        Transitions from 'ingesting', 'processing', or 'completed' to 'ingested'.
        This should only be called AFTER the data has been successfully written.

        Args:
            unit_id: ID of work unit to mark as ingested
            max_retries: Maximum number of retry attempts for database locks
        """
        import random

        for attempt in range(max_retries):
            try:
                with self._connect(timeout=60.0) as conn:
                    cursor = conn.execute(
                        """
                        UPDATE work_units
                        SET status = 'ingested',
                            completed_at = ?
                        WHERE unit_id = ?
                          AND status IN ('ingesting', 'processing', 'completed')
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

                    conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    base_delay = 0.1 * (2 ** attempt)
                    jitter = random.uniform(0, base_delay * 0.5)
                    time.sleep(base_delay + jitter)
                    continue
                import sys
                # If we can't mark a unit as ingested, raise exception - unit would be re-ingested
                raise RuntimeError(
                    f"Failed to mark unit {unit_id} as ingested after {max_retries} attempts: {e}"
                ) from e

    def reset_incomplete_ingestions(self, output_dir: Path) -> int:
        """
        Reset units marked as 'ingesting' or 'ingested' that still have shards on disk.

        This recovery method handles cases where:
        - Ingest reader claimed a unit but writer crashed before processing
        - Writer processed but crashed before marking as ingested
        - Pipeline was interrupted during ingestion

        Units with shards on disk are reset to 'completed' so they can be re-ingested.

        Args:
            output_dir: Directory containing shard databases

        Returns:
            Number of units reset
        """
        import os

        reset_count = 0
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
            # Find all units marked as ingesting or ingested
            cursor = conn.execute(
                """
                SELECT unit_id
                FROM work_units
                WHERE status IN ('ingesting', 'ingested')
                """
            )
            units = [row[0] for row in cursor.fetchall()]

            # Check which ones still have shards on disk
            for unit_id in units:
                shard_path = output_dir / f"{unit_id}.db"
                if shard_path.exists():
                    # Shard still exists - reset to completed for re-ingestion
                    conn.execute(
                        """
                        UPDATE work_units
                        SET status = 'completed'
                        WHERE unit_id = ?
                        """,
                        (unit_id,)
                    )
                    reset_count += 1

            if reset_count > 0:
                conn.commit()

        return reset_count

    def clear_all_work_units(self) -> None:
        """Clear all work units from the database."""
        with self._connect(timeout=30.0) as conn:
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
        with self._connect(timeout=30.0) as conn:
            # First, mark any processing units that have children as completed
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
            cursor = conn.execute(
                """
                UPDATE work_units
                SET status = 'pending',
                    claimed_by = NULL,
                    claimed_at = NULL
                WHERE status = 'processing'
                """
            )
            conn.commit()
            return cursor.rowcount

    def reset_all_work_units(self) -> int:
        """
        Reset ALL work units (including completed ones) back to pending.

        This is used in reprocess mode to re-run all work from scratch
        while preserving the work unit partitions/boundaries.

        Returns:
            Number of units reset
        """
        with self._connect(timeout=30.0) as conn:
            cursor = conn.execute(
                """
                UPDATE work_units
                SET status = 'pending',
                    claimed_by = NULL,
                    claimed_at = NULL
                WHERE status IN ('completed', 'processing')
                """
            )
            conn.commit()
            return cursor.rowcount
