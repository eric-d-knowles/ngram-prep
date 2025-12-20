"""Database schema and connection management for work tracking."""

from __future__ import annotations

import sqlite3
from pathlib import Path

__all__ = ["init_tracking_database"]


def init_tracking_database(db_path: Path) -> None:
    """
    Initialize the work tracking database schema.

    Creates tables for work units and metadata if they don't exist.

    Args:
        db_path: Path to SQLite database file
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(db_path), timeout=10.0) as conn:
        # Enable WAL mode for better concurrency
        # WAL allows multiple readers while one writer is active
        conn.execute("PRAGMA journal_mode=WAL")

        # Increase cache size for better performance
        conn.execute("PRAGMA cache_size=10000")

        # Use NORMAL synchronous mode (safer than OFF, faster than FULL)
        conn.execute("PRAGMA synchronous=NORMAL")

        # Work units table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS work_units (
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

        # Index on status for efficient queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status
            ON work_units(status)
        """)

        # Metadata table for tracking global state
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        conn.commit()
