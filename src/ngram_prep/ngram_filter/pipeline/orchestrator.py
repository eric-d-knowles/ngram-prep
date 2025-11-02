"""Main orchestrator for the ngram filter pipeline."""

from __future__ import annotations

import multiprocessing as mp
import shutil
import sys
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Optional

from setproctitle import setproctitle

from ..config import PipelineConfig, FilterConfig
from ngram_prep.tracking import (
    create_uniform_work_units,
    create_smart_work_units,
    WorkTracker,
    SimpleOutputManager,
)
from ngram_prep.utilities.display import truncate_path_to_fit
from ngram_prep.common_db import open_db
from .simple_worker_pool import run_worker_pool_simple
from .parallel_ingest import ingest_shards_parallel
from .worker import WorkerConfig
from .progress import (
    create_counters,
    print_phase_banner,
    run_progress_reporter,
)
from .whitelist import write_whitelist
from .setup import PipelineSetup
from .display import (
    print_pipeline_header,
    print_phase_header,
    print_completion_banner,
)

__all__ = ["build_processed_db", "PipelineOrchestrator"]


class PipelineOrchestrator:
    """Orchestrates the complete ngram filtering pipeline."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        filter_config: FilterConfig,
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            pipeline_config: Configuration for pipeline execution
            filter_config: Configuration for ngram filtering
        """
        self.pipeline_config = pipeline_config
        self.filter_config = filter_config

        # Initialize setup helper
        self.setup = PipelineSetup(pipeline_config, filter_config)
        self.temp_paths = self.setup.temp_paths

        self.total_items = 0
        self.total_bytes = 0
        self.output_whitelist_path = getattr(
            pipeline_config,
            "output_whitelist_path",
            None,
        )

    def run(self) -> None:
        """Execute the complete pipeline."""
        setproctitle("ngf:orchestrator")

        # Print configuration header
        worker_config = self._create_worker_config()
        print_pipeline_header(
            self.pipeline_config,
            self.filter_config,
            worker_config,
            self.temp_paths,
        )

        # Prepare all resources using setup module
        self.temp_paths, self.filter_config = self.setup.prepare_all()

        # Execute pipeline phases
        self._create_work_units()
        self._process_work_units()
        self._ingest_shards()
        self._finalize_database()
        self._compact_if_requested()
        self._validate_final_result()
        self._generate_output_whitelist()
        self._print_completion_banner()

    def _create_work_units(self) -> None:
        """Create or resume work units for processing."""
        print_phase_header(1, "Creating work units...")

        work_tracker = WorkTracker(
            self.temp_paths['work_tracker'],
            claim_order=self.pipeline_config.work_unit_claim_order
        )
        num_workers = self.pipeline_config.num_workers

        mode = getattr(self.pipeline_config, 'mode', 'resume')

        if mode == 'restart':
            print("Clean restart - creating new work units")
            work_tracker.clear_all_work_units()
            self._create_new_work_units(work_tracker, num_workers)

        elif mode == 'reprocess':
            print("Reprocess - creating new work units and resetting status")
            work_tracker.clear_all_work_units()
            self._create_new_work_units(work_tracker, num_workers)

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
            print(f"Sampling database to create {num_work_units} density-based work units...")

            # Get sampling parameters from config
            num_sampling_workers = self.pipeline_config.num_sampling_workers
            if num_sampling_workers is None:
                num_sampling_workers = min(num_work_units, 40)
            samples_per_worker = self.pipeline_config.samples_per_worker

            try:
                work_units = create_smart_work_units(
                    db_path=self.pipeline_config.src_db,
                    num_units=num_work_units,
                    num_sampling_workers=num_sampling_workers,
                    samples_per_worker=samples_per_worker,
                    read_profile=self.pipeline_config.reader_profile
                )
                print(f"Created {len(work_units)} balanced work units based on data density")
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

        # Skip processing if no pending work units
        if progress.pending == 0:
            print("No pending work units - skipping processing phase")
            return

        # Set up progress monitoring
        progress_reporter = self._setup_progress_monitoring()

        try:
            # Run the worker pool (workers write shards, no concurrent ingestion)
            run_worker_pool_simple(
                num_workers=num_workers,
                src_db_path=self.temp_paths['src_db'],
                work_tracker_path=self.temp_paths['work_tracker'],
                output_dir=self.temp_paths['output_dir'],
                filter_config=self.filter_config,
                pipeline_config=self.pipeline_config,
                worker_config=self._create_worker_config(),
                counters=(
                    progress_reporter['counters']
                    if progress_reporter
                    else None
                ),
            )

            print(f"{'─' * 37} final {'─' * 37}")

        finally:
            # Stop progress reporter
            self._cleanup_progress_monitoring(progress_reporter)

    def _setup_progress_monitoring(self) -> Optional[dict]:
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
            name="ngf:reporter"
        )
        reporter_process.start()

        return {
            'counters': counters,
            'process': reporter_process,
            'stop_event': stop_event
        }

    def _cleanup_progress_monitoring(
        self,
        progress_reporter: Optional[dict],
    ) -> None:
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

    def _ingest_shards(self) -> None:
        """Ingest all shards with parallel reads and sequential writes."""
        # Check if ingestion is already complete
        work_tracker = WorkTracker(
            self.temp_paths['work_tracker'],
            claim_order=self.pipeline_config.work_unit_claim_order
        )
        progress = work_tracker.get_progress()

        # Get shard count for header
        import sqlite3
        with sqlite3.connect(str(self.temp_paths['work_tracker']), timeout=10.0) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM work_units WHERE status = 'completed'")
            num_shards = cursor.fetchone()[0]

        num_readers = getattr(self.pipeline_config, "ingest_num_readers", 4)
        queue_size = getattr(self.pipeline_config, "ingest_queue_size", 8)

        print_phase_header(3, f"Ingesting {num_shards} shards with {num_readers} parallel readers...")

        # Skip ingestion if all shards are already ingested
        if progress.ingested == progress.total and num_shards == 0:
            print("All shards already ingested - skipping ingestion phase")
            return

        total_items, total_bytes = ingest_shards_parallel(
            output_dir=self.temp_paths['output_dir'],
            dst_db_path=self.temp_paths['dst_db'],
            work_tracker_path=self.temp_paths['work_tracker'],
            pipeline_config=self.pipeline_config,
            num_readers=num_readers,
            queue_size=queue_size,
        )

        self.total_items = total_items
        self.total_bytes = total_bytes

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

        from ngram_prep.utilities.display import format_bytes, format_banner
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

    def _generate_output_whitelist(self) -> None:
        """Generate whitelist from final merged database if requested."""
        if not self.output_whitelist_path:
            return

        output_whitelist_path = Path(self.output_whitelist_path)
        output_top_n = getattr(
            self.pipeline_config,
            "output_whitelist_top_n",
            None,
        )
        spell_check = getattr(
            self.pipeline_config,
            "output_whitelist_spell_check",
            False,
        )
        spell_check_language = getattr(
            self.pipeline_config,
            "output_whitelist_spell_check_language",
            "en_US",
        )

        print_phase_header(5, "Generating output whitelist...")

        prefix = "  Output path: "
        path = truncate_path_to_fit(output_whitelist_path, prefix)
        print(f"{prefix}{path}")

        if output_top_n:
            print(f"  Extracting top {output_top_n:,} tokens")
        else:
            print("  Extracting all tokens")

        if spell_check:
            print(f"  Spell checking enabled ({spell_check_language})")

        start_time = time.perf_counter()
        final_path = write_whitelist(
            db_or_path=self.temp_paths['dst_db'],
            dest=output_whitelist_path,
            top=output_top_n,
            decode=True,
            sep="\t",
            spell_check=spell_check,
            spell_check_language=spell_check_language,
        )
        elapsed = time.perf_counter() - start_time

        # Count lines to report token count
        try:
            with final_path.open('r', encoding='utf-8') as f:
                token_count = sum(1 for _ in f)
            print(
                f"  Generated whitelist with {token_count:,} tokens "
                f"in {elapsed:.1f}s"
            )
        except Exception:
            print(f"  Whitelist generated in {elapsed:.1f}s")

    def _print_completion_banner(self) -> None:
        """Print completion banner."""
        print_completion_banner(
            dst_db_path=self.temp_paths['dst_db'],
            total_items=self.total_items,
            total_bytes=self.total_bytes,
            output_whitelist_path=self.output_whitelist_path,
        )

    def _create_worker_config(self) -> WorkerConfig:
        """
        Create worker configuration from pipeline config.

        Returns:
            Configured WorkerConfig instance
        """
        return WorkerConfig(
            disable_wal=getattr(
                self.pipeline_config,
                'writer_disable_wal',
                True,
            ),
            disable_compaction=True,
        )


def build_processed_db(
    pipeline_config: PipelineConfig,
    filter_config: FilterConfig,
) -> None:
    """
    Main entry point for the ngram filtering pipeline.

    Args:
        pipeline_config: Configuration for pipeline execution
        filter_config: Configuration for ngram filtering
    """
    orchestrator = PipelineOrchestrator(pipeline_config, filter_config)
    orchestrator.run()
