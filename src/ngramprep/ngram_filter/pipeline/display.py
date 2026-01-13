"""Display and formatting utilities for the ngram filter pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ngramprep.utilities.display import truncate_path_to_fit, format_bytes, format_banner
from ngramprep.utilities.progress import ProgressDisplay
from ..config import PipelineConfig, FilterConfig
from .worker import WorkerConfig

__all__ = [
    "print_pipeline_header",
    "print_phase_header",
    "print_completion_banner",
]

# Shared display instance
_display = ProgressDisplay(width=100)


def print_pipeline_header(
    pipeline_config: PipelineConfig,
    filter_config: FilterConfig,
    worker_config: WorkerConfig,
    temp_paths: dict[str, Path],
) -> None:
    """
    Print pipeline configuration summary.

    Args:
        pipeline_config: Pipeline configuration
        filter_config: Filter configuration
        worker_config: Worker configuration
        temp_paths: Dictionary of pipeline paths
    """
    from datetime import datetime

    LINE_WIDTH = 100
    start_time = datetime.now()

    # Get configuration values
    mode = pipeline_config.mode
    compact_after_ingest = getattr(pipeline_config, "compact_after_ingest", False)
    num_workers = pipeline_config.num_workers
    num_initial_work_units = getattr(pipeline_config, 'num_initial_work_units', None) or num_workers
    read_profile = pipeline_config.reader_profile
    write_profile = pipeline_config.writer_profile
    flush_interval = getattr(pipeline_config, 'flush_interval_s', 10.0)
    num_ingest_readers = getattr(pipeline_config, "ingest_num_readers", 4)
    ingest_queue_size = getattr(pipeline_config, "ingest_queue_size", 8)

    # Format paths
    src_path_str = truncate_path_to_fit(temp_paths['src_db'], "Source DB:            ")
    dst_path_str = truncate_path_to_fit(temp_paths['dst_db'], "Target DB:            ")
    tmp_path_str = truncate_path_to_fit(temp_paths['tmp_dir'], "Temp directory:       ")

    # Get bin size from filter config
    bin_size = getattr(filter_config, "bin_size", 1)
    bin_desc = "annual data" if bin_size == 1 else f"{bin_size}-year bins"

    lines = [
        "",
        "N-GRAM FILTER PIPELINE",
        "━" * LINE_WIDTH,
        f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}",
        f"Mode:       {mode.upper()}",
        "",
        "Configuration",
        "═" * LINE_WIDTH,
        f"Source DB:            {src_path_str}",
        f"Target DB:            {dst_path_str}",
        f"Temp directory:       {tmp_path_str}",
        "",
        "Parallelism",
        "─" * LINE_WIDTH,
        f"Workers:              {num_workers}",
        f"Initial work units:   {num_initial_work_units}",
        "",
        "Database Profiles",
        "─" * LINE_WIDTH,
        f"Reader profile:       {read_profile}",
        f"Writer profile:       {write_profile}",
        "",
        "Ingestion Configuration",
        "─" * LINE_WIDTH,
        f"Ingest readers:       {num_ingest_readers}",
        f"Queue size:           {ingest_queue_size}",
        f"Compact after ingest: {compact_after_ingest}",
        "",
        "Worker Configuration",
        "─" * LINE_WIDTH,
        f"Flush interval:       {flush_interval}s",
        "",
        "Temporal Binning",
        "─" * LINE_WIDTH,
        f"Bin size:             {bin_size} ({bin_desc})",
        "",
    ]

    # Add filter options
    filter_lines = _format_filter_options(filter_config)
    if filter_lines:
        lines.extend([
            "Filter Options",
            "─" * LINE_WIDTH,
        ])
        lines.extend(filter_lines)
        lines.append("")

    # Add whitelist information
    whitelist_lines = _format_whitelist_info(filter_config, pipeline_config)
    if whitelist_lines:
        lines.extend([
            "Whitelist Configuration",
            "─" * LINE_WIDTH,
        ])
        lines.extend(whitelist_lines)
        lines.append("")

    print("\n".join(lines), flush=True)


def _format_filter_options(filter_config: FilterConfig) -> list[str]:
    """Format filter options information.

    Args:
        filter_config: Filter configuration

    Returns:
        List of formatted lines for filter options
    """
    lines = []

    # Check if whitelist is being used
    whitelist_path = getattr(filter_config, "whitelist_path", None)
    if whitelist_path:
        lines.append(f"Whitelist mode:       ENABLED")
        lines.append(f"(All other filters are bypassed when whitelist is active)")
        return lines

    # Normalization options
    lowercase = getattr(filter_config, "lowercase", True)
    lines.append(f"Lowercase:            {lowercase}")

    alpha_only = getattr(filter_config, "alpha_only", True)
    lines.append(f"Alpha only:           {alpha_only}")

    ascii_alpha_only = getattr(filter_config, "ascii_alpha_only", False)
    lines.append(f"ASCII alpha only:     {ascii_alpha_only}")

    # N-gram context requirement
    min_context_tokens = getattr(filter_config, "min_context_tokens", 2)
    lines.append(f"Min context tokens:   {min_context_tokens}")

    # Length filtering
    filter_short = getattr(filter_config, "filter_short", False)
    if filter_short:
        min_len = getattr(filter_config, "min_len", 3)
        lines.append(f"Min token length:     {min_len}")
    else:
        lines.append(f"Min token length:     disabled")

    # Stopword filtering
    filter_stops = getattr(filter_config, "filter_stops", True)
    stop_set = getattr(filter_config, "stop_set", None)
    if filter_stops and stop_set:
        lines.append(f"Stopword filtering:   enabled ({len(stop_set)} stopwords)")
    elif filter_stops:
        lines.append(f"Stopword filtering:   enabled (no stopwords loaded)")
    else:
        lines.append(f"Stopword filtering:   disabled")

    # Lemmatization
    apply_lemmatization = getattr(filter_config, "apply_lemmatization", True)
    lemma_gen = getattr(filter_config, "lemma_gen", None)
    if apply_lemmatization and lemma_gen:
        lines.append(f"Lemmatization:        enabled")
    elif apply_lemmatization:
        lines.append(f"Lemmatization:        enabled (no lemmatizer loaded)")
    else:
        lines.append(f"Lemmatization:        disabled")

    # Always-include tokens
    always_include = getattr(filter_config, "always_include", None)
    if always_include:
        lines.append(f"Always include:       {len(always_include)} token(s)")
    else:
        lines.append(f"Always include:       none")

    return lines


def _format_whitelist_info(filter_config: FilterConfig, pipeline_config: PipelineConfig) -> list[str]:
    """Format whitelist configuration information.

    Args:
        filter_config: Filter configuration
        pipeline_config: Pipeline configuration

    Returns:
        List of formatted lines for whitelist configuration
    """
    lines = []

    # Input whitelist info
    input_whitelist_path = getattr(filter_config, "whitelist_path", None)
    if input_whitelist_path:
        path_str = truncate_path_to_fit(input_whitelist_path, "Input whitelist:      ")
        lines.append(f"Input whitelist:      {path_str}")

        min_count = getattr(filter_config, "whitelist_min_count", 1)
        top_n = getattr(filter_config, "whitelist_top_n", None)
        if top_n:
            lines.append(f"  Top {top_n:,} tokens (min count: {min_count})")
        else:
            lines.append(f"  All tokens (min count: {min_count})")
    else:
        lines.append("Input whitelist:      None")

    # Output whitelist info
    output_whitelist_path = getattr(pipeline_config, "output_whitelist_path", None)
    if output_whitelist_path:
        path_str = truncate_path_to_fit(output_whitelist_path, "Output whitelist:     ")
        lines.append(f"Output whitelist:     {path_str}")

        output_top_n = getattr(pipeline_config, "output_whitelist_top_n", None)
        if output_top_n:
            lines.append(f"  Top {output_top_n:,} tokens")
        else:
            lines.append("  All tokens")

        # Add spell check info
        spell_check = getattr(pipeline_config, "output_whitelist_spell_check", False)
        if spell_check:
            spell_check_language = getattr(pipeline_config, "output_whitelist_spell_check_language", "en_US")
            lines.append(f"  Spell checking enabled ({spell_check_language})")

        # Add year range info
        year_range = getattr(pipeline_config, "output_whitelist_year_range", None)
        if year_range:
            start_year, end_year = year_range
            lines.append(f"  Year range: {start_year}-{end_year} (inclusive)")
    else:
        lines.append("Output whitelist:     None")

    return lines


def print_phase_header(phase_num: int, description: str) -> None:
    """
    Print a phase header.

    Args:
        phase_num: Phase number
        description: Phase description
    """
    _display.print_banner(f"Phase {phase_num}: {description}", include_blank=True)


def print_completion_banner(
    dst_db_path: Path,
    total_items: int,
    total_bytes: int,
    output_whitelist_path: Optional[Path] = None,
) -> None:
    """
    Print completion banner with final statistics.

    Args:
        dst_db_path: Path to destination database
        total_items: Total items in final database
        total_bytes: Total bytes in final database
        output_whitelist_path: Optional path to output whitelist
    """
    # Format the database size using format_bytes utility
    size_formatted = format_bytes(total_bytes)

    # Box is 100 chars wide, content area is 96 chars (100 - 2 borders - 2 padding)
    # Each line formatted as " key: value " so available = 96 - 2 (spaces) = 94
    content_width = 94

    # Truncate paths to fit within box content area
    db_path_truncated = truncate_path_to_fit(dst_db_path, "Database: ", total_width=content_width)

    # Build summary items
    summary_items = {
        "Items": f"{total_items:,} (estimated)",
        "Size": size_formatted,
        "Database": db_path_truncated,
    }

    if output_whitelist_path:
        whitelist_truncated = truncate_path_to_fit(output_whitelist_path, "Whitelist: ", total_width=content_width)
        summary_items["Whitelist"] = whitelist_truncated

    # Print using ProgressDisplay summary box
    print()  # Blank line before
    _display.print_summary_box(
        title="PROCESSING COMPLETE",
        items=summary_items,
        box_width=100,
    )
