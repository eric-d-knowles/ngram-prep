# ngram_filter/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Dict, Any, Union


# Filtering options used by filter processor
@dataclass(frozen=True)
class FilterConfig:
    lowercase: bool = True
    alpha_only: bool = True
    filter_short: bool = True
    filter_stops: bool = True
    apply_lemmatization: bool = True
    min_len: int = 3
    stop_set: Optional[Set[str]] = None
    lemma_gen: Any = None
    tag_map: Optional[Dict] = None
    vocab_path: Optional[Union[str, Path]] = None
    vocab_view: Any = None

    # Whitelist configuration (for input filtering)
    whitelist_path: Optional[Union[str, Path]] = None
    whitelist_min_count: int = 1
    whitelist_top_n: Optional[int] = None
    whitelist: Optional[Set[bytes]] = None

    # Always-include tokens configuration (tokens to preserve regardless of whitelist)
    always_include: Optional[Set[bytes]] = None  # Set of tokens to always include (e.g., b"working-class", b"nuclear")

    # Year binning configuration
    bin_size: int = 1  # Aggregate years into bins (1 = annual data, 5 = 5-year bins, etc.)


# Pipeline orchestration options
@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline orchestration configuration for parallel filter execution.

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
    flush_interval_s: float = 10.0  # How often to flush buffer

    # Work unit partitioning
    use_smart_partitioning: bool = True  # Use density-based partitioning (slower startup, better balance)
    num_sampling_workers: Optional[int] = None  # Parallel workers for sampling (default: min(num_units, 40))
    samples_per_worker: int = 50_000  # Reservoir size per sampling worker (higher = more accurate)
    cache_partitions: bool = True  # Cache smart partitioning results to avoid re-sampling
    use_cached_partitions: bool = True  # Use cached partitions if available (otherwise resample)

    # Progress reporting
    progress_every_s: float = 60.0

    # Reader/Writer profiles
    reader_profile: str = "read:packed24"
    writer_profile: str = 'write:packed24'
    writer_disable_wal: bool = True

    # Ingest configuration (parallel reads, sequential writes)
    # Note: Shards are always deleted after successful ingestion
    ingest_read_profile: str = 'read:packed24'
    ingest_write_profile: str = 'write:packed24'
    ingest_batch_items: int = 2_000_000
    ingest_disable_wal: bool = True
    ingest_num_readers: int = 8  # Number of parallel shard reader processes
    ingest_queue_size: int = 8  # Max shards buffered in memory (controls memory usage)

    # Pipeline control
    mode: str = "resume"  # "restart" (wipe all), "resume" (continue), or "reprocess" (wipe DB, keep cache)
    compact_after_ingest: bool = True
    work_unit_claim_order: str = "random"  # "sequential" or "random"

    # Output whitelist generation (from filtered results)
    output_whitelist_path: Optional[Union[str, Path]] = None
    output_whitelist_top_n: Optional[int] = None
    output_whitelist_spell_check: bool = False  # Only include correctly spelled words
    output_whitelist_spell_check_language: str = "en_US"  # Language for spell checking
    output_whitelist_year_range: Optional[tuple[int, int]] = None  # (start_year, end_year) - only include ngrams present in all years in range