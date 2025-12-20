"""Shared types for parallel processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["WorkUnit", "WorkProgress"]


@dataclass
class WorkUnit:
    """Represents a unit of work (key range) to process."""

    unit_id: str
    """Unique identifier for this work unit"""

    start_key: Optional[bytes]
    """Starting key (inclusive), None for beginning of keyspace"""

    end_key: Optional[bytes]
    """Ending key (exclusive), None for end of keyspace"""

    parent_id: Optional[str] = None
    """ID of parent unit if this was created by splitting"""

    current_position: Optional[bytes] = None
    """Current scan position for tracking progress (used for incremental splits)"""


@dataclass
class WorkProgress:
    """Progress statistics for work units."""

    total: int
    pending: int
    processing: int
    completed: int
    failed: int
    saved: int = 0  # No longer used (inline ingestion)
    ingesting: int = 0  # Units claimed for ingestion but not yet written
    ingested: int = 0
    split: int = 0  # Total number of split operations performed
    splitting: int = 0  # Deprecated: No longer used (no 'split' status)
    active_workers: int = 0  # Number of distinct workers currently processing units
    idle_workers: int = 0  # Number of workers not currently processing (based on total - active)
    starving_workers: int = 0  # Number of workers idle for >= starvation_threshold checks
