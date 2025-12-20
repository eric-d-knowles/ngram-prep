"""Main orchestrator for the ngram pivot pipeline."""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from datetime import timedelta
from pathlib import Path

from setproctitle import setproctitle

from ..config import PipelineConfig
from ngramprep.tracking import (
    create_uniform_work_units,
    create_smart_work_units,
    WorkTracker,
    PartitionCache,
    PartitionCacheKey,
)
from ngramprep.common_db import open_db
from .worker_pool import run_worker_pool
from .worker import WorkerConfig
from .progress import (
    create_counters,
    print_phase_banner,
    run_progress_reporter,
)
from .setup import PipelineSetup
from .display import (
    print_pipeline_header,
    print_phase_header,
    print_completion_banner,
)

__all__ = ["run_pivot_pipeline", "build_pivoted_db", "PivotOrchestrator"]


class PivotOrchestrator:
    """Orchestrates the complete ngram pivot pipeline."""

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize pipeline orchestrator.

        Args:
            pipeline_config: Configuration for pipeline execution
        """
        self.pipeline_config = pipeline_config

        # Initialize setup helper
        self.setup = PipelineSetup(pipeline_config)
        self.temp_paths = {}

        self.total_items = 0
        self.total_bytes = 0

    def run(self) -> None:
        """Execute the complete pipeline."""
        setproctitle("ngp:orchestrator")

        # Print configuration header
        worker_config = self._create_worker_config()

        # Prepare all resources using setup module
        self.temp_paths = self.setup.prepare_all()

        print_pipeline_header(
            self.pipeline_config,
            worker_config,
            self.temp_paths,
        )

        # Execute pipeline phases
        self._create_work_units()
        self._process_work_units()
        self._ingest_shards()
        self._finalize_database()
        self._compact_if_requested()
        self._validate_final_result()
        self._print_completion_banner()

    def _create_work_units(self) -> None:
        """Create or resume work units for processing."""
        print_phase_header(1, "Creating work units...")

        work_tracker = WorkTracker(
            self.temp_paths['work_tracker'],
            claim_order=self.pipeline_config.work_unit_claim_order
        )
        num_workers = self.pipeline_config.num_workers

        mode = self.pipeline_config.mode

        if mode == 'restart':
            print("Clean restart - creating new work units")
            work_tracker.clear_all_work_units()
            self._create_new_work_units(work_tracker, num_workers)

        elif mode == 'reprocess':
            print("Reprocess - using cached partitions if available")
            # In reprocess mode, we try to use cached partitions
            # If cache exists, reset all work units to pending
            # If no cache, create new work units
            progress = work_tracker.get_progress()
            if progress.total == 0:
                print("No existing work units - creating new ones")
                self._create_new_work_units(work_tracker, num_workers)
            else:
                print("Found existing work units - resetting all to pending")
                work_tracker.reset_all_work_units()

        elif mode == 'resume':
            progress = work_tracker.get_progress()
            if progress.total == 0:
                print("No existing work units - creating new ones")
                self._create_new_work_units(work_tracker, num_workers)
            else:
                print("Resuming existing work units")
                self._resume_existing_work_units(work_tracker, progress)

        else:
            raise ValueError(
                f"Invalid mode: {mode}. "
                f"Must be 'restart', 'resume', or 'reprocess'"
            )

    def _create_new_work_units(
        self,
        work_tracker: WorkTracker,
        num_workers: int,
    ) -> None:
        """
        Create work units using density-based or uniform partitioning.

        For large databases, uses parallel sampling to create balanced partitions.
        Falls back to uniform partitioning for small datasets.

        Args:
            work_tracker: WorkTracker instance
            num_workers: Number of workers (used if num_initial_work_units not set)
        """
        # Use num_initial_work_units if specified, otherwise default to num_workers
        num_work_units = getattr(
            self.pipeline_config,
            'num_initial_work_units',
            None
        ) or num_workers

        # Check if smart partitioning is enabled and appropriate
        use_smart_partitioning = self.pipeline_config.use_smart_partitioning

        # For very small unit counts, uniform is fine
        if num_work_units < 4:
            use_smart_partitioning = False

        if use_smart_partitioning:
            # Get sampling parameters from config
            num_sampling_workers = self.pipeline_config.num_sampling_workers
            if num_sampling_workers is None:
                num_sampling_workers = min(num_work_units, 40)
            samples_per_worker = self.pipeline_config.samples_per_worker

            # Check if we should use cached partitions
            cache_enabled = getattr(self.pipeline_config, 'use_cached_partitions', True)
            should_cache = getattr(self.pipeline_config, 'cache_partitions', True)

            # Initialize partition cache
            partition_cache = PartitionCache(self.temp_paths['base'])
            cache_key = PartitionCacheKey(
                db_path=str(self.pipeline_config.src_db),
                num_units=num_work_units,
                samples_per_worker=samples_per_worker,
                num_sampling_workers=num_sampling_workers
            )

            # Try to load from cache first
            work_units = None
            if cache_enabled and partition_cache.exists(cache_key):
                print(f"Loading cached partitions ({num_work_units} work units)...")
                try:
                    work_units = partition_cache.load(cache_key)
                    if work_units:
                        print(f"Loaded {len(work_units)} work units from cache")
                except Exception as e:
                    print(f"Failed to load cached partitions: {e}")
                    print(f"Will resample database...")
                    work_units = None

            # If no cached partitions, sample the database
            if work_units is None:
                print(f"Sampling database to create {num_work_units} density-based work units...")
                try:
                    work_units = create_smart_work_units(
                        db_path=self.pipeline_config.src_db,
                        num_units=num_work_units,
                        num_sampling_workers=num_sampling_workers,
                        samples_per_worker=samples_per_worker,
                        read_profile=self.pipeline_config.reader_profile
                    )
                    print(f"Created {len(work_units)} balanced work units based on data density")

                    # Cache the results if enabled
                    if should_cache:
                        try:
                            partition_cache.save(cache_key, work_units)
                            print(f"Cached partition results for future use")
                        except Exception as e:
                            print(f"Warning: Failed to cache partitions: {e}")

                except Exception as e:
                    print(f"Smart partitioning failed: {e}")
                    print(f"Falling back to uniform partitioning...")
                    work_units = create_uniform_work_units(num_work_units)
                    print(f"Created {len(work_units)} uniform work units (byte-range partitioning)")
        else:
            # Create uniform work units (fast, no sampling needed)
            work_units = create_uniform_work_units(num_work_units)
            print(f"Created {len(work_units)} uniform work units (byte-range partitioning)")

        work_tracker.add_work_units(work_units)

    def _resume_existing_work_units(
        self,
        work_tracker: WorkTracker,
        progress,
    ) -> None:
        """
        Resume processing existing work units.

        Args:
            work_tracker: WorkTracker instance
            progress: Progress information
        """
        print(
            f"Resuming: {progress.completed} completed, "
            f"{progress.processing} processing, {progress.pending} pending"
        )

        # Reset processing units from interrupted run
        if progress.processing > 0:
            work_tracker.reset_all_processing_units()

    def _process_work_units(self) -> None:
        """Process work units using worker pool."""
        work_tracker = WorkTracker(
            self.temp_paths['work_tracker'],
            claim_order=self.pipeline_config.work_unit_claim_order
        )
        progress = work_tracker.get_progress()
        num_workers = self.pipeline_config.num_workers

        print_phase_header(
            2,
            f"Processing {progress.pending} work units with {num_workers} workers..."
        )

        # Set up progress monitoring
        progress_reporter = self._setup_progress_monitoring()

        try:
            # Run the worker pool (workers write shards, no concurrent ingestion)
            run_worker_pool(
                num_workers=num_workers,
                src_db_path=self.temp_paths['src_db'],
                work_tracker_path=self.temp_paths['work_tracker'],
                output_dir=self.temp_paths['output_dir'],
                pipeline_config=self.pipeline_config,
                worker_config=self._create_worker_config(),
                counters=(
                    progress_reporter['counters']
                    if progress_reporter
                    else None
                ),
            )

            print(f"{'─' * 45} final {'─' * 44}")

        finally:
            # Stop progress reporter
            self._cleanup_progress_monitoring(progress_reporter)

    def _ingest_shards(self) -> None:
        """Ingest all shards with parallel reads and sequential writes."""
        enable_ingest = getattr(self.pipeline_config, 'enable_ingest', True)
        if not enable_ingest:
            print("Skipping ingestion (enable_ingest=False)")
            return

        # Reset incomplete ingestions from previous interrupted run
        # This recovers units stuck in 'ingesting' state (claimed but not written)
        work_tracker = WorkTracker(
            self.temp_paths['work_tracker'],
            claim_order=self.pipeline_config.work_unit_claim_order
        )
        reset_count = work_tracker.reset_incomplete_ingestions(self.temp_paths['output_dir'])
        if reset_count > 0:
            print(f"Recovered {reset_count} incomplete ingestion(s) from previous run")

        # Get shard count for header
        import sqlite3
        with sqlite3.connect(str(self.temp_paths['work_tracker']), timeout=10.0) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM work_units WHERE status = 'completed'")
            num_shards = cursor.fetchone()[0]

        if num_shards == 0:
            print("No shards to ingest (all already ingested)")
            return

        # Route to appropriate ingestion strategy based on ingest_mode
        ingest_mode = getattr(self.pipeline_config, "ingest_mode", "write_batch")

        if ingest_mode == "direct_sst":
            # Direct SST ingestion (fast path for non-overlapping key ranges)
            print_phase_header(3, f"Ingesting {num_shards} SST files via direct ingestion...")
            from .worker_pool import ingest_coordinator_sst_direct
            ingest_coordinator_sst_direct(
                dst_db_path=self.temp_paths['dst_db'],
                work_tracker_path=self.temp_paths['work_tracker'],
                output_dir=self.temp_paths['output_dir'],
                pipeline_config=self.pipeline_config,
            )
        else:
            # Write batch ingestion (default, works for all cases)
            num_readers = getattr(self.pipeline_config, "num_ingest_readers", 8)
            buffer_shards = getattr(self.pipeline_config, "ingest_buffer_shards", 3)
            total_buffered = num_readers * buffer_shards

            print_phase_header(3, f"Ingesting {num_shards} shards with {num_readers} workers (each buffers {buffer_shards} shards = {total_buffered} total in memory)...")

            from .worker_pool import ingest_coordinator_process
            ingest_coordinator_process(
                dst_db_path=self.temp_paths['dst_db'],
                work_tracker_path=self.temp_paths['work_tracker'],
                output_dir=self.temp_paths['output_dir'],
                pipeline_config=self.pipeline_config,
                stop_event=None,  # Separate-stage mode - reader stops when no more units
            )

    def _setup_progress_monitoring(self):
        """
        Set up progress monitoring if enabled.

        Returns:
            Dictionary with monitoring resources or None if disabled
        """
        if self.pipeline_config.progress_every_s <= 0:
            return None

        ctx = mp.get_context("spawn")
        counters = create_counters(ctx)
        start_time = time.perf_counter()
        num_workers = self.pipeline_config.num_workers

        print_phase_banner()
        stop_event = ctx.Event()

        reporter_process = ctx.Process(
            target=run_progress_reporter,
            args=(
                counters,
                start_time,
                self.pipeline_config.progress_every_s,
                stop_event,
                num_workers,
                self.temp_paths['work_tracker'],
            ),
            daemon=True,
            name="ngp:reporter"
        )
        reporter_process.start()

        return {
            'counters': counters,
            'process': reporter_process,
            'stop_event': stop_event
        }

    def _cleanup_progress_monitoring(self, progress_reporter) -> None:
        """
        Clean up progress monitoring resources.

        Args:
            progress_reporter: Monitoring resources or None
        """
        if not progress_reporter:
            return

        try:
            progress_reporter['stop_event'].set()
            progress_reporter['process'].join(timeout=2)
        except Exception:
            pass

    def _finalize_database(self) -> None:
        """Finalize database after ingestion."""
        print_phase_header(4, "Finalizing database...")

        # Get stats from existing DB
        try:
            with open_db(self.temp_paths['dst_db'], mode="r") as db:
                prop = db.get_property("rocksdb.estimate-num-keys")
                self.total_items = int(prop) if prop else 0
                prop = db.get_property("rocksdb.total-sst-files-size")
                self.total_bytes = int(prop) if prop else 0
        except Exception:
            self.total_items = 0
            self.total_bytes = 0

        # Perform final flush
        with open_db(self.temp_paths['dst_db'], mode="rw") as db:
            try:
                db.finalize_bulk()
            except Exception:
                pass

    def _compact_if_requested(self) -> None:
        """Perform compaction on the final database if requested."""
        compact_after_ingest = getattr(
            self.pipeline_config,
            'compact_after_ingest',
            False,
        )

        if not compact_after_ingest:
            return

        from ngramprep.utilities.display import format_bytes, format_banner
        import logging

        logger = logging.getLogger(__name__)

        logger.info("Starting post-ingestion compaction")
        print()
        print(format_banner("Post-Ingestion Compaction", style="─"))

        # Open database for compaction with the same profile used for writing
        write_profile = getattr(
            self.pipeline_config,
            'ingest_write_profile',
            'write:packed24',
        )
        with open_db(self.temp_paths['dst_db'], profile=write_profile) as db:
            # Get initial size if possible
            try:
                initial_size = db.get_property("rocksdb.total-sst-files-size")
                initial_size = int(initial_size) if initial_size else None
                if initial_size:
                    print(f"Initial DB size:         {format_bytes(initial_size)}")
            except Exception:
                initial_size = None

            sys.stdout.flush()

            start_time = time.time()
            try:
                db.compact_all()
                elapsed = time.time() - start_time

                print(f"Compaction completed in {timedelta(seconds=int(elapsed))}")

                # Get final size if possible
                try:
                    final_size = db.get_property("rocksdb.total-sst-files-size")
                    final_size = int(final_size) if final_size else None
                    if initial_size and final_size:
                        saved = initial_size - final_size
                        pct = (saved / initial_size) * 100
                        print(f"Size before:             {format_bytes(initial_size)}")
                        print(f"Size after:              {format_bytes(final_size)}")
                        print(f"Space saved:             {format_bytes(saved)} ({pct:.1f}%)")
                        # Update total_bytes to reflect post-compaction size
                        self.total_bytes = final_size
                except Exception:
                    pass

            except Exception as e:
                logger.error(f"Compaction failed: {e}")
                print(f"Compaction failed: {e}")
                print("Database is still usable, but may not be optimally compacted.")

    def _validate_final_result(self) -> None:
        """Validate the final merged database."""
        if not self.temp_paths['dst_db'].exists():
            raise RuntimeError("Final database was not created")

        # Check for failed work units
        work_tracker = WorkTracker(
            self.temp_paths['work_tracker'],
            claim_order=self.pipeline_config.work_unit_claim_order
        )
        progress = work_tracker.get_progress()

        if progress.failed > 0:
            raise RuntimeError(
                f"Pipeline completed with {progress.failed} failed work units! "
                f"Data is incomplete. Check logs for errors."
            )

        if progress.pending > 0 or progress.processing > 0:
            raise RuntimeError(
                f"Pipeline completed but {progress.pending} units still pending "
                f"and {progress.processing} still processing. This indicates a bug."
            )

        if progress.ingested != progress.total:
            raise RuntimeError(
                f"Pipeline completed but only {progress.ingested}/{progress.total} "
                f"units were ingested. Data is incomplete."
            )

        try:
            with open_db(self.temp_paths['dst_db'], mode="r") as result_db:
                result_db.get_property("rocksdb.estimate-num-keys")
        except Exception as e:
            print(f"Could not validate final database: {e}")

    def _print_completion_banner(self) -> None:
        """Print completion banner."""
        print_completion_banner(
            dst_db_path=self.temp_paths['dst_db'],
            total_items=self.total_items,
            total_bytes=self.total_bytes,
        )

    def _create_worker_config(self) -> WorkerConfig:
        """
        Create worker configuration from pipeline config.

        Returns:
            Configured WorkerConfig instance
        """
        return WorkerConfig(
            disable_wal=self.pipeline_config.writer_disable_wal,
            disable_compaction=True,
        )


def _setup_logging(pipeline_config: PipelineConfig) -> None:
    """
    Set up logging for the pivot pipeline if not already configured.

    Args:
        pipeline_config: Pipeline configuration
    """
    import logging

    # Check if logging is already configured
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        # Logging already configured, don't override
        return

    # Set up logging to temp directory
    from ngramprep.ngram_acquire.logger import setup_logger
    from datetime import datetime

    log_dir = pipeline_config.tmp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = setup_logger(
        db_path=str(log_dir),
        filename_prefix="pivot_pipeline",
        console=False,
        rotate=True,
        max_bytes=100_000_000,
        backup_count=5,
        force=False  # Don't override if already configured
    )

    # Log initial context
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Pivot Pipeline")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Source DB: {pipeline_config.src_db}")
    logger.info(f"Destination DB: {pipeline_config.dst_db}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)


def run_pivot_pipeline(pipeline_config: PipelineConfig) -> None:
    """
    Main entry point for the ngram pivot pipeline.

    Args:
        pipeline_config: Configuration for pipeline execution
    """
    # Set up logging if not already configured
    _setup_logging(pipeline_config)

    orchestrator = PivotOrchestrator(pipeline_config)
    orchestrator.run()


def build_pivoted_db(
    pipeline_config: Optional[PipelineConfig] = None,
    # Path construction parameters (used if pipeline_config not provided)
    ngram_size: Optional[int] = None,
    repo_release_id: Optional[str] = None,
    repo_corpus_id: Optional[str] = None,
    db_path_stub: Optional[str] = None,
    # Pipeline execution parameters
    num_workers: Optional[int] = None,
    mode: Optional[str] = None,
    # Work unit partitioning
    use_smart_partitioning: Optional[bool] = None,
    num_initial_work_units: Optional[int] = None,
    cache_partitions: Optional[bool] = None,
    use_cached_partitions: Optional[bool] = None,
    samples_per_worker: Optional[int] = None,
    work_unit_claim_order: Optional[str] = None,
    # Progress and flushing
    flush_interval_s: Optional[float] = None,
    progress_every_s: Optional[float] = None,
    # Ingestion configuration
    enable_ingest: Optional[bool] = None,
    num_ingest_readers: Optional[int] = None,
    ingest_buffer_shards: Optional[int] = None,
    ingest_mode: Optional[str] = None,
    compact_after_ingest: Optional[bool] = None,
    # Database profiles
    reader_profile: Optional[str] = None,
    writer_disable_wal: Optional[bool] = None,
) -> str:
    """
    Convenience wrapper for the ngram pivot pipeline.

    Can be called in two ways:
    1. With a PipelineConfig object (for advanced usage)
    2. With path stub parameters (convenience wrapper like download_and_ingest_to_rocksdb)

    Args:
        pipeline_config: Pre-configured PipelineConfig (if provided with paths, other parameters are ignored)
        ngram_size: N-gram size (1-5) - required if pipeline_config not provided
        repo_release_id: Release date in YYYYMMDD format (e.g., "20200217") - required if pipeline_config not provided
        repo_corpus_id: Corpus identifier (e.g., "eng", "eng-us") - required if pipeline_config not provided
        db_path_stub: Base directory for database (will be expanded) - required if pipeline_config not provided
        num_workers: Number of parallel workers (default: cpu_count() - 1 or num_initial_work_units, whichever is lower)
        mode: "restart" (wipe all), "resume" (continue), or "reprocess" (wipe DB, keep cache)
        use_smart_partitioning: Use density-based partitioning for better load balancing
        num_initial_work_units: Initial number of work units (default: num_workers)
        cache_partitions: Cache smart partitioning results
        use_cached_partitions: Use cached partitions if available
        samples_per_worker: Reservoir size per sampling worker
        work_unit_claim_order: "sequential" or "random"
        flush_interval_s: How often to flush buffer and check for splits
        progress_every_s: Progress reporting interval
        enable_ingest: Whether to ingest shards into final DB
        num_ingest_readers: Number of parallel shard reader processes
        ingest_buffer_shards: Number of shards buffered per reader
        ingest_mode: "write_batch" or "direct_sst"
        compact_after_ingest: Perform full compaction after ingestion
        reader_profile: Database read profile
        writer_disable_wal: Disable WAL for writers

    Returns:
        Path to the pivoted database directory
    """
    from typing import Optional

    # Check if we have a complete pipeline_config with paths
    has_complete_pipeline_config = (
        pipeline_config is not None and
        hasattr(pipeline_config, 'src_db') and
        hasattr(pipeline_config, 'dst_db') and
        hasattr(pipeline_config, 'tmp_dir') and
        pipeline_config.src_db is not None and
        pipeline_config.dst_db is not None and
        pipeline_config.tmp_dir is not None
    )

    # If we have a complete pipeline_config with all required paths, use it directly
    if has_complete_pipeline_config:
        run_pivot_pipeline(pipeline_config)
        return str(pipeline_config.dst_db)

    # Otherwise, we need path stub parameters to construct the config
    if ngram_size is None or repo_release_id is None or repo_corpus_id is None or db_path_stub is None:
        raise ValueError(
            "Either provide a complete pipeline_config with paths (src_db, dst_db, tmp_dir), "
            "or provide all path stub parameters (ngram_size, repo_release_id, repo_corpus_id, db_path_stub)"
        )

    from ngramprep.ngram_acquire.db.build_path import build_db_path

    # Construct paths using same logic as download function
    base_path = Path(build_db_path(db_path_stub, ngram_size, repo_release_id, repo_corpus_id)).parent

    # Construct derived paths
    src_db = base_path / f"{ngram_size}grams_processed.db"
    dst_db = base_path / f"{ngram_size}grams_pivoted.db"
    tmp_dir = base_path / "pivoting_tmp"

    # Determine default num_workers if not specified
    if num_workers is None:
        import os
        cpu_count = os.cpu_count() or 1
        default_workers = max(1, cpu_count - 1)
        if num_initial_work_units is not None:
            num_workers = min(default_workers, num_initial_work_units)
        else:
            num_workers = default_workers

    # Create pipeline config, only passing non-None parameters to use PipelineConfig defaults
    config_kwargs = {
        'src_db': src_db,
        'dst_db': dst_db,
        'tmp_dir': tmp_dir,
        'num_workers': num_workers,
    }

    # Add optional parameters only if explicitly provided
    optional_params = {
        'mode': mode,
        'use_smart_partitioning': use_smart_partitioning,
        'num_initial_work_units': num_initial_work_units,
        'cache_partitions': cache_partitions,
        'use_cached_partitions': use_cached_partitions,
        'samples_per_worker': samples_per_worker,
        'work_unit_claim_order': work_unit_claim_order,
        'flush_interval_s': flush_interval_s,
        'progress_every_s': progress_every_s,
        'enable_ingest': enable_ingest,
        'num_ingest_readers': num_ingest_readers,
        'ingest_buffer_shards': ingest_buffer_shards,
        'ingest_mode': ingest_mode,
        'compact_after_ingest': compact_after_ingest,
        'reader_profile': reader_profile,
        'writer_disable_wal': writer_disable_wal,
    }

    for key, value in optional_params.items():
        if value is not None:
            config_kwargs[key] = value

    constructed_pipeline_config = PipelineConfig(**config_kwargs)

    # Run the pipeline
    run_pivot_pipeline(constructed_pipeline_config)

    return str(dst_db)
