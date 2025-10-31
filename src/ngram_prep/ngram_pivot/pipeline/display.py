"""Display formatting for the pivot pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ngram_prep.utilities.display import truncate_path_to_fit, format_bytes
from ..config import PipelineConfig
from .worker import WorkerConfig

__all__ = [
    "print_pipeline_header",
    "print_phase_header",
    "print_completion_banner",
]

LINE_WIDTH = 100


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
        f"Split check interval: {pipeline_config.split_check_interval_s}s",
        "",
        "Database Profiles",
        "─" * LINE_WIDTH,
        f"Reader profile:       {pipeline_config.reader_profile}",
        f"Writer profile:       {pipeline_config.writer_profile}",
        f"Ingest profile:       {pipeline_config.ingest_write_profile}",
        "",
        "Flush Configuration",
        "─" * LINE_WIDTH,
        f"Flush interval:       {getattr(pipeline_config, 'flush_interval_s', 5.0)}s",
        "",
    ]
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
    from datetime import datetime

    end_time = datetime.now()

    dst_path_str = truncate_path_to_fit(dst_db_path, "Output database:      ")

    lines = [
        "",
        "Pipeline Complete",
        "═" * LINE_WIDTH,
        f"Output database:      {dst_path_str}",
        f"Total records:        {total_items:,}",
        f"Database size:        {format_bytes(total_bytes)}",
        f"End Time:             {end_time:%Y-%m-%d %H:%M:%S}",
        "━" * LINE_WIDTH,
        "",
    ]
    print("\n".join(lines), flush=True)
