"""Display formatting for the pivot pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ngramprep.utilities.display import truncate_path_to_fit, format_bytes
from ngramprep.utilities.progress import ProgressDisplay
from ..config import PipelineConfig
from .worker import WorkerConfig

__all__ = [
    "print_pipeline_header",
    "print_phase_header",
    "print_completion_banner",
]

LINE_WIDTH = 100

# Shared display instance
_display = ProgressDisplay(width=100)


def print_pipeline_header(
    pipeline_config: PipelineConfig,
    worker_config: WorkerConfig,
    temp_paths: dict,
) -> None:
    """
    Print the pipeline configuration header.

    Args:
        pipeline_config: Pipeline configuration
        worker_config: Worker configuration
        temp_paths: Dictionary of temporary paths
    """
    from datetime import datetime

    start_time = datetime.now()

    # Format paths
    src_path_str = truncate_path_to_fit(pipeline_config.src_db, "Source DB:            ")
    dst_path_str = truncate_path_to_fit(pipeline_config.dst_db, "Target DB:            ")
    tmp_path_str = truncate_path_to_fit(pipeline_config.tmp_dir, "Temp directory:       ")

    # Get configuration values
    ingest_mode = getattr(pipeline_config, 'ingest_mode', 'write_batch')
    num_ingest_readers = getattr(pipeline_config, 'num_ingest_readers', 8)
    ingest_buffer_shards = getattr(pipeline_config, 'ingest_buffer_shards', 3)
    compact_after_ingest = getattr(pipeline_config, 'compact_after_ingest', True)

    lines = [
        "",
        "PARALLEL N-GRAM DATABASE PIVOT",
        "━" * LINE_WIDTH,
        f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}",
        f"Mode:       {pipeline_config.mode.upper()}",
        "",
        "Configuration",
        "═" * LINE_WIDTH,
        f"Source DB:            {src_path_str}",
        f"Target DB:            {dst_path_str}",
        f"Temp directory:       {tmp_path_str}",
        "",
        "Parallelism",
        "─" * LINE_WIDTH,
        f"Workers:              {pipeline_config.num_workers}",
        f"Initial work units:   {pipeline_config.num_initial_work_units or pipeline_config.num_workers}",
        "",
        "Database Profiles",
        "─" * LINE_WIDTH,
        f"Reader profile:       {pipeline_config.reader_profile}",
        f"Writer profile:       {pipeline_config.writer_profile}",
        f"Ingest profile:       {pipeline_config.ingest_write_profile}",
        "",
        "Ingestion Configuration",
        "─" * LINE_WIDTH,
        f"Ingest mode:          {ingest_mode}",
        f"Ingest readers:       {num_ingest_readers}",
    ]

    # Add buffer info only for write_batch mode
    if ingest_mode == "write_batch":
        lines.append(f"Buffer shards/reader: {ingest_buffer_shards}")

    lines.extend([
        f"Compact after ingest: {compact_after_ingest}",
        "",
        "Worker Configuration",
        "─" * LINE_WIDTH,
        f"Flush interval:       {getattr(pipeline_config, 'flush_interval_s', 5.0)}s",
        "",
    ])
    print("\n".join(lines), flush=True)


def print_phase_header(phase: int, title: str) -> None:
    """
    Print a phase header.

    Args:
        phase: Phase number
        title: Phase title
    """
    print(f"\nPhase {phase}: {title}")
    print("═" * LINE_WIDTH)


def print_completion_banner(
    dst_db_path: Path,
    total_items: int,
    total_bytes: int,
) -> None:
    """
    Print completion banner with statistics.

    Args:
        dst_db_path: Path to destination database
        total_items: Total number of items written
        total_bytes: Total bytes written
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

    # Print using ProgressDisplay summary box
    print()  # Blank line before
    _display.print_summary_box(
        title="PROCESSING COMPLETE",
        items=summary_items,
        box_width=100,
    )
