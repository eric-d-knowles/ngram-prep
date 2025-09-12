# ngram_filter/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Dict, Any, Union


# Filtering options used by readers.build_processor(...)
@dataclass(frozen=True)
class FilterConfig:
    opt_lower: bool = True
    opt_alpha: bool = True
    opt_shorts: bool = True
    opt_stops: bool = True
    opt_lemmas: bool = True
    min_len: int = 3
    stop_set: Optional[Set[str]] = None
    lemma_gen: Any = None
    tag_map: Optional[Dict] = None
    vocab_path: Optional[Union[str, Path]] = None
    vocab_view: Any = None


# Pipeline orchestration options
@dataclass(frozen=True)
class PipelineConfig:
    # I/O
    src_db: Path
    dst_db: Path
    tmp_dir: Path

    # Parallelism
    readers: int = 8

    # Progress reporting
    progress_every_s: float = 10.0

    # Adaptive batch sizing with min/max bounds
    min_items_per_bucket: int = 1_000
    max_items_per_bucket: int = 25_000
    min_bytes_per_bucket: int = 2 * 1024 ** 2
    max_bytes_per_bucket: int = 16 * 1024 ** 2

    # Writer performance optimizations
    writer_size_check_interval: int = 100
    writer_batch_prealloc: int = 2000
    writer_target_latency_ms: float = 50.0

    # Reader configuration
    reader_slices: int = 0
    reader_slice_factor: int = 16
    reader_slice_prefix_len: int = 0

    # Queue management
    queue_size_multiplier: float = 1.0
    max_queue_size: int = 50_000

    # Stage A (source reading)
    source_read_profile: str = "read:packed24"

    # Stage B (sharded writers)
    writer_profile: str = "write:packed24"
    writer_disable_wal: bool = True

    # Stage C (final ingest/merge)
    ingest_read_profile: str = "read:packed24"
    ingest_write_profile: str = "write:packed24"
    ingest_batch_bytes: int = 128 << 20
    ingest_batch_items: int = 200_000
    ingest_disable_wal: bool = True
    ingest_diag_every_batches: int = 25
    ingest_diag_every_seconds: float = 3.0
    finalize_shards: bool = True

    # Memory management
    enable_memory_monitoring: bool = True
    memory_limit_gb: Optional[float] = None
    gc_frequency: int = 1000

    # Performance tuning
    use_xxhash: bool = True
    enable_adaptive_batching: bool = True
    enable_process_titles: bool = True

    force_restart: bool = False
