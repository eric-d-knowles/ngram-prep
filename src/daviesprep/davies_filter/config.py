# daviesprep/davies_filter/config.py
"""Configuration for Davies corpus filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Any, List, Union


@dataclass(frozen=True)
class FilterConfig:
    """
    Configuration for Davies sentence filtering.

    Similar to ngram FilterConfig but simpler (no POS tags).
    """
    lowercase: bool = True
    alpha_only: bool = True
    ascii_alpha_only: bool = True
    filter_short: bool = True
    filter_stops: bool = True
    apply_lemmatization: bool = True
    min_context_tokens: int = 2
    min_len: int = 3
    stop_set: Optional[Set[str]] = None
    stop_words_language: Optional[str] = None  # Language code for stopwords (e.g., "russian", "english")
    lemma_gen: Any = None  # Lemmatizer instance
    whitelist: Optional[Set[bytes]] = None  # Whitelist of allowed tokens (bytes)
    always_include: Optional[Set[bytes]] = None  # Set internally by pipeline; not for direct use


@dataclass(frozen=True)
class PipelineConfig:
    """
    Pipeline orchestration configuration for parallel filter execution.

    Path construction:
        Provide db_path_stub (e.g., "/path/to/COHA/"). The corpus name is derived
        from the directory name. Source and destination databases are auto-derived:
        - Source: {db_path_stub}/{corpus_name} (or {corpus_name}_{genre} with genre_focus)
        - Dest:   {db_path_stub}/{corpus_name}_filtered

    Example::

        pipeline_config = PipelineConfig(
            db_path_stub="/scratch/NLP_corpora/COHA/",
            num_workers=24,
            compact_after=True,
            output_whitelist_top_n=30_000,
            output_whitelist_year_range=(1950, 2019),
            output_whitelist_spell_check=True,
        )
    """
    # Path construction
    db_path_stub: Optional[Union[str, Path]] = None
    genre_focus: Optional[List[str]] = None

    # Processing
    num_workers: int = 8
    batch_size: int = 50_000

    # DB profiles
    reader_profile: str = "read:packed24"
    writer_profile: str = "write:packed24"

    # Pipeline control
    compact_after: bool = False

    # Output whitelist generation (from filtered results)
    output_whitelist_path: Optional[Union[str, Path]] = "default"  # "default" = auto-derive from db_path_stub
    output_whitelist_top_n: Optional[int] = None  # If None, skip whitelist generation
    output_whitelist_year_range: Optional[tuple[int, int]] = None
    output_whitelist_spell_check: bool = False
    output_whitelist_spell_check_language: str = "en_US"
    output_whitelist_apply: bool = True  # After generating whitelist, apply it as a second filter pass
    output_whitelist_workers: Optional[int] = None  # Defaults to num_workers
    output_whitelist_batch_size: int = 50_000

    # Tokens to always preserve regardless of whitelist (e.g., gendered pronouns)
    always_include: Optional[Set[Union[str, bytes]]] = None
