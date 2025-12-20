# ngram_pivot/config.py
"""Configuration for pivot operations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline orchestration configuration for parallel pivot execution.

    Mode options:
        - "restart": Wipe output DB and cache, create fresh work units
        - "resume": Continue from last checkpoint (preserves all state)
        - "reprocess": Wipe output DB but reuse cached partitions
    """

    # I/O
    src_db: Path
    dst_db: Path
    tmp_dir: Path

    # Parallelism
    num_workers: int = 8
    num_initial_work_units: Optional[int] = None  # If None, defaults to num_workers
    flush_interval_s: float = 5.0  # How often to flush buffer and check for splits (0 to disable)

    # Work unit partitioning
    use_smart_partitioning: bool = True  # Use density-based partitioning (slower startup, better balance)
    num_sampling_workers: Optional[int] = None  # Parallel workers for sampling (default: min(num_units, 40))
    samples_per_worker: int = 50_000  # Reservoir size per sampling worker (higher = more accurate)
    cache_partitions: bool = True  # Cache smart partitioning results to avoid re-sampling
    use_cached_partitions: bool = True  # Use cached partitions if available (otherwise resample)

    # Progress reporting
    progress_every_s: float = 5.0

    # Reader/Writer profiles
    reader_profile: str = "read:packed24"
    writer_profile: str = 'write:packed24'
    writer_disable_wal: bool = True

    # Ingest configuration (runs as separate stage after workers complete)
    ingest_mode: Literal["write_batch", "direct_sst"] = "direct_sst"  # Ingestion strategy
    ingest_read_profile: str = 'read:packed24'
    ingest_write_profile: str = 'write:packed24'
    ingest_disable_wal: bool = True
    num_ingest_readers: int = 8  # Number of parallel reader/writer processes for ingestion
    ingest_buffer_shards: int = 3  # Number of shards each worker reads and buffers before writing (higher = more parallel I/O)

    # Pipeline control
    mode: Literal["restart", "resume", "reprocess"] = "resume"
    enable_ingest: bool = True  # If False, skip ingestion stage (shards remain on disk)
    compact_after_ingest: bool = True
    work_unit_claim_order: Literal["sequential", "random"] = "sequential"

    # Validation
    validate: bool = True
