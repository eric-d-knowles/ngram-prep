# daviesprep/davies_filter/config.py
"""Configuration for Davies corpus filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Any, List


@dataclass(frozen=True)
class FilterConfig:
    """
    Configuration for Davies sentence filtering.

    Similar to ngram FilterConfig but simpler (no POS tags).
    """
    lowercase: bool = True
    alpha_only: bool = True
    filter_short: bool = True
    filter_stops: bool = True
    apply_lemmatization: bool = True
    min_len: int = 3
    stop_set: Optional[Set[str]] = None
    lemma_gen: Any = None  # Lemmatizer instance
    whitelist: Optional[Set[bytes]] = None  # Whitelist of allowed tokens (bytes)
    always_include: Optional[Set[bytes]] = None  # Tokens to always preserve (e.g., b"working-class")


@dataclass(frozen=True)
class PipelineConfig:
    """
    Pipeline orchestration configuration for parallel filter execution.

    Simplified version of ngram PipelineConfig for Davies data.
    """
    # I/O
    src_db: Path
    dst_db: Path

    # Processing
    num_workers: int = 8
    write_batch_size: int = 100_000

    # DB profiles
    reader_profile: str = "read:packed24"
    writer_profile: str = "write:packed24"
    writer_disable_wal: bool = True

    # Pipeline control
    overwrite_db: bool = True

    # Genre filtering
    genre_focus: Optional[List[str]] = None  # If set, only process these genres
