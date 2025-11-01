"""Display and formatting utilities for the ngram filter pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ngram_prep.utilities.display import truncate_path_to_fit, format_bytes, format_banner
from ngram_prep.utilities.progress import ProgressDisplay
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
    print(format_banner("N-GRAM FILTER PIPELINE", width=100, style="━"))

    mode = pipeline_config.mode
    compact_after_ingest = getattr(pipeline_config, "compact_after_ingest", False)
    num_workers = pipeline_config.num_workers

    _display.print_section("Configuration", style="═")

    print("\033[4mPipeline\033[0m")
    pipeline_items = {
        "Run mode": mode,
    }
    if compact_after_ingest:
        pipeline_items["Compact after ingest"] = True
    _display.print_config_items(pipeline_items)

    print("\n\033[4mWorkers\033[0m")
    read_profile = pipeline_config.reader_profile
    write_profile = pipeline_config.writer_profile
    flush_interval = getattr(pipeline_config, 'flush_interval_s', 5.0)

    num_initial_work_units = getattr(pipeline_config, 'num_initial_work_units', None) or num_workers

    _display.print_config_items({
        "Num Workers": num_workers,
        "Initial work units": num_initial_work_units,
        "Profiles": f"read={read_profile}, write={write_profile}",
        "Flush interval": f"{flush_interval}s",
    })

    print("\n\033[4mFiles\033[0m")
    prefix = "Source: "
    src_path = truncate_path_to_fit(temp_paths['src_db'], prefix)
    print(f"{prefix}{src_path}")

    prefix = "Destination: "
    dst_path = truncate_path_to_fit(temp_paths['dst_db'], prefix)
    print(f"{prefix}{dst_path}")

    # Input whitelist info
    _print_input_whitelist_info(filter_config)

    # Output whitelist info
    output_whitelist_path = getattr(
        pipeline_config,
        "output_whitelist_path",
        None,
    )
    if output_whitelist_path:
        output_top_n = getattr(
            pipeline_config,
            "output_whitelist_top_n",
            None,
        )
        prefix = "Output whitelist: "
        suffix = f" (top {output_top_n:,} keys)"
        whitelist_path = truncate_path_to_fit(
            output_whitelist_path,
            prefix + suffix,
        )
        print(f"{prefix}{whitelist_path}{suffix}")
    else:
        print("Output whitelist: None")


def _print_input_whitelist_info(filter_config: FilterConfig) -> None:
    """Print input whitelist configuration."""
    whitelist_path = getattr(filter_config, "whitelist_path", None)
    if whitelist_path:
        prefix = "Input whitelist: "
        path = truncate_path_to_fit(whitelist_path, prefix)
        print(f"{prefix}{path}")

        min_count = getattr(filter_config, "whitelist_min_count", 1)
        top_n = getattr(filter_config, "whitelist_top_n", None)
        if top_n:
            print(f"  Top {top_n:,} tokens (min count: {min_count})")
        else:
            print(f"  All tokens (min count: {min_count})")
    else:
        print("Input whitelist: None")


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
        "Items": f"{total_items:,}",
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
