"""Progress tracking and statistics for work units."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from .types import WorkProgress

__all__ = ["ProgressTracker"]


class ProgressTracker:
    """Tracks progress statistics for work units."""

    def __init__(self, db_path: Path):
        """
        Initialize the progress tracker.

        Args:
            db_path: Path to SQLite tracking database
        """
        self.db_path = Path(db_path)

    def _connect(self, timeout: float = 30.0) -> sqlite3.Connection:
        """
        Create a database connection with WAL mode enabled.

        Args:
            timeout: Database lock timeout in seconds

        Returns:
            SQLite connection
        """
        conn = sqlite3.connect(str(self.db_path), timeout=timeout)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get_progress(self, num_workers: Optional[int] = None) -> WorkProgress:
        """
        Get current progress statistics.

        Args:
            num_workers: Total number of workers (optional, used to calculate idle_workers)

        Returns:
            WorkProgress with current counts and active worker count
        """
        with self._connect(timeout=30.0) as conn:
            # Count units that represent active/actual work:
            # - Leaf units (units not split): all statuses
            # - Parent units with status='completed'/'ingesting'/'ingested' (finished processing)
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
                    -- Include parent units that are processing, completed, ingesting, or ingested
                    -- (parent units with status='processing' are still being actively worked on)
                    (status IN ('processing', 'completed', 'ingesting', 'ingested')
                     AND unit_id IN (
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
                WHERE status = 'processing'
                  AND claimed_by IS NOT NULL
                """
            )
            active_workers = cursor.fetchone()[0]

            # Calculate idle workers if total worker count provided
            idle_workers = (num_workers - active_workers) if num_workers is not None else 0

            # Get starving worker count from metadata
            cursor = conn.execute(
                """
                SELECT value
                FROM metadata
                WHERE key = 'starving_workers'
                """
            )
            row = cursor.fetchone()
            starving_workers = int(row[0]) if row else 0

            return WorkProgress(
                total=total,
                pending=counts.get('pending', 0),
                processing=counts.get('processing', 0),
                completed=counts.get('completed', 0),
                failed=counts.get('failed', 0),
                saved=0,  # No longer used: inline ingestion
                ingesting=counts.get('ingesting', 0),
                ingested=counts.get('ingested', 0),
                split=0,  # Dynamic splitting removed
                splitting=0,  # No longer used (no 'split' status)
                active_workers=active_workers,
                idle_workers=idle_workers,
                starving_workers=starving_workers,
            )
