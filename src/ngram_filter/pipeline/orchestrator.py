"""Main orchestrator for the ngram filter pipeline."""

from __future__ import annotations

import multiprocessing as mp
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

from setproctitle import setproctitle

from ..config import PipelineConfig, FilterConfig
from .work_tracker import WorkTracker, create_work_units, validate_work_units
from .worker import WorkerConfig, run_worker_pool
from .ingest import ingest_shards_streaming
from .progress import create_counters, print_phase_banner, run_progress_reporter
from common_db.api import open_db
from .whitelist import load_whitelist, write_whitelist


class PipelineOrchestrator:
    """Orchestrates the complete ngram filtering pipeline."""

    def __init__(self, pipeline_config: PipelineConfig, filter_config: FilterConfig):
        self.pipeline_config = pipeline_config
        self.filter_config = filter_config
        self.temp_paths = self._initialize_paths()

        self.total_items = 0
        self.total_bytes = 0
        self.output_whitelist_path = None

    def _initialize_paths(self) -> dict[str, Path]:
        """Initialize and prepare all required paths."""
        paths = {
            'src_db': self.pipeline_config.src_db.resolve(),
            'dst_db': self.pipeline_config.dst_db.resolve(),
            'tmp_dir': self.pipeline_config.tmp_dir.resolve(),
        }

        paths['work_tracker'] = paths['tmp_dir'] / "work_tracker.db"
        paths['output_dir'] = paths['tmp_dir'] / "worker_outputs"

        return paths

    def run(self) -> None:
        """Execute the complete pipeline."""
        setproctitle("ngf:orchestrator")

        self._print_pipeline_header()
        self._prepare_directories()
        self._prepare_vocabulary_index()
        self._prepare_whitelist()

        # Execute pipeline phases
        self._create_work_units()
        self._process_work_units()
        self._merge_results()
        self._validate_final_result()
        self._generate_output_whitelist()
        self._print_done_banner()

    def _print_pipeline_header(self) -> None:
        """Print pipeline configuration summary."""
        print("N-GRAM FILTER PIPELINE")
        print("━" * 100)

        num_workers = getattr(self.pipeline_config, 'readers', 16)
        num_work_units = num_workers * 8

        worker_config = self._create_worker_config()

        if len(str(self.temp_paths['src_db'])) > 75:
            src_db_str = str(self.temp_paths['src_db'])[-72:]
        else:
            src_db_str = str(self.temp_paths['src_db'])

        if len(str(self.temp_paths['dst_db'])) > 75:
            dst_db_str = str(self.temp_paths['dst_db'])[-72:]
        else:
            dst_db_str = str(self.temp_paths['dst_db'])

        print("\nConfiguration:")
        print("═" * 100)
        print(f"  Workers: {num_workers}")
        print(f"  Work units: {num_work_units}")
        print(f"  Source: {src_db_str}")
        print(f"  Destination: {dst_db_str}")
        print(f"  Buffer: {worker_config.buffer_size:,} items, "
              f"{worker_config.buffer_bytes // (1024 * 1024)}MB")
        print(f"  Profile: {worker_config.profile}")

        # Add input whitelist info
        whitelist_path = getattr(self.filter_config, "whitelist_path", None)
        if whitelist_path:
            if len(str(whitelist_path)) > 75:
                whitelist_path_str = "..." + str(whitelist_path)[-72:]
            print(f"  Input whitelist: {whitelist_path_str}")
            min_count = getattr(self.filter_config, "whitelist_min_count", 1)
            top_n = getattr(self.filter_config, "whitelist_top_n", None)
            if top_n:
                print(f"    Top {top_n:,} tokens (min count: {min_count})")
            else:
                print(f"    All tokens (min count: {min_count})")
        else:
            print(f"  Input whitelist: None")

        # Add output whitelist info and store in instance variable
        self.output_whitelist_path = getattr(self.pipeline_config, "output_whitelist_path", None)
        if self.output_whitelist_path:
            output_top_n = getattr(self.pipeline_config, "output_whitelist_top_n", None)
            if len(str(self.output_whitelist_path)) > 75:
                output_whitelist_path_str = "..." + str(self.output_whitelist_path)[-59:]
            print(f"  Output whitelist: {output_whitelist_path_str} (top {output_top_n:,} keys)")
        else:
            print(f"  Output whitelist: None")

    def _prepare_directories(self) -> None:
        """Clean and create necessary directories."""
        # Clean up destination (always safe to recreate)
        if self.temp_paths['dst_db'].exists():
            shutil.rmtree(self.temp_paths['dst_db'])
        self.temp_paths['dst_db'].parent.mkdir(parents=True, exist_ok=True)

        # For temp directory: only clean if force_restart is enabled
        force_restart = getattr(self.pipeline_config, 'force_restart', False)

        if force_restart and self.temp_paths['tmp_dir'].exists():
            shutil.rmtree(self.temp_paths['tmp_dir'])

        # Create temp directories if they don't exist
        self.temp_paths['tmp_dir'].mkdir(parents=True, exist_ok=True)
        self.temp_paths['output_dir'].mkdir(exist_ok=True)

    def _prepare_vocabulary_index(self) -> None:
        """Prepare memory-mapped vocabulary index if needed."""
        vocab_path = getattr(self.filter_config, "vocab_path", None)
        if not vocab_path:
            return

        from ngram_filter.filters.shared_vocab import build_vocab_index

        idx_prefix = vocab_path.parent / "vocab_mmap"
        idx_file = idx_prefix.with_suffix(".idx")
        lex_file = idx_prefix.with_suffix(".lex")

        # Check if index needs rebuilding
        if self._vocabulary_index_needs_rebuild(vocab_path, idx_file, lex_file):
            print("Building vocabulary index...")
            build_vocab_index(vocab_path, idx_prefix)

        # Update filter config to use the index
        self.filter_config = replace(self.filter_config, vocab_path=idx_prefix)

    def _vocabulary_index_needs_rebuild(
            self, vocab_path: Path, idx_file: Path, lex_file: Path
    ) -> bool:
        """Check if vocabulary index needs to be rebuilt."""
        if not idx_file.exists() or not lex_file.exists():
            return True

        # Check if vocab file is newer than index files
        vocab_mtime = vocab_path.stat().st_mtime
        idx_mtime = max(idx_file.stat().st_mtime, lex_file.stat().st_mtime)
        return vocab_mtime > idx_mtime

    def _prepare_whitelist(self) -> None:
        """Load whitelist if specified in filter config."""
        whitelist_path = getattr(self.filter_config, "whitelist_path", None)
        if not whitelist_path:
            return

        whitelist_path = Path(whitelist_path)
        if not whitelist_path.exists():
            raise FileNotFoundError(f"Whitelist file not found: {whitelist_path}")

        print(f"Loading whitelist from {whitelist_path}...")

        # Load whitelist with optional parameters from config
        min_count = getattr(self.filter_config, "whitelist_min_count", 1)
        top_n = getattr(self.filter_config, "whitelist_top_n", None)

        whitelist = load_whitelist(
            whitelist_path=whitelist_path,
            min_count=min_count,
            top_n=top_n,
        )

        print(f"  Loaded {len(whitelist):,} tokens")

        # Store the loaded whitelist in filter_config for workers to use
        self.filter_config = replace(self.filter_config, whitelist=whitelist)

    def _create_work_units(self) -> None:
        """Create or resume work units for processing."""
        print("\nPhase 1: Creating work units...")
        print("═" * 100)

        work_tracker = WorkTracker(self.temp_paths['work_tracker'])
        num_workers = getattr(self.pipeline_config, 'readers', 16)
        num_work_units = num_workers * 8

        force_restart = getattr(self.pipeline_config, 'force_restart', False)

        if force_restart:
            work_tracker.clear_all_work_units()
            self._create_new_work_units(work_tracker, num_work_units)
        else:
            progress = work_tracker.get_progress()
            if progress.total == 0:
                # No existing work units found - create new ones
                self._create_new_work_units(work_tracker, num_work_units)
            else:
                # Resume existing work units
                self._resume_existing_work_units(work_tracker, progress)

    def _create_new_work_units(self, work_tracker: WorkTracker, num_work_units: int) -> None:
        """Create new work units from scratch."""
        force_restart = getattr(self.pipeline_config, 'force_restart', False)
        if force_restart:
            print(f"  Force restart requested - clearing existing work units")
            work_tracker.clear_all_work_units()

        # print("  Creating new work units...")
        work_units = create_work_units(self.temp_paths['src_db'], num_work_units)

        # print("  Validating work units...")
        if not validate_work_units(self.temp_paths['src_db'], work_units):
            print("  WARNING: Work unit validation failed - proceeding anyway")
            print("  This may indicate work unit ranges don't align with your data")
        # else:
        #    print("  Work units validated successfully")

        work_tracker.add_work_units(work_units)
        progress = work_tracker.get_progress()
        print(f"  Created {len(work_units)} work units")

    def _resume_existing_work_units(self, work_tracker: WorkTracker, progress) -> None:
        """Resume processing existing work units."""
        print(f"  Resuming: {progress.completed} completed, "
              f"{progress.processing} processing, {progress.pending} pending")

        # Reset all processing units on resume (they're from an interrupted run)
        if progress.processing > 0:
            reset_count = work_tracker.reset_all_processing_units()

            # Show updated progress
            updated_progress = work_tracker.get_progress()

    def _process_work_units(self) -> None:
        """Process work units using worker pool."""
        work_tracker = WorkTracker(self.temp_paths['work_tracker'])
        progress = work_tracker.get_progress()
        num_workers = getattr(self.pipeline_config, 'readers', 16)

        print(f"\nPhase 2: Processing {progress.pending} work units with {num_workers} workers...")
        print("═" * 100)

        # Set up progress monitoring
        progress_reporter = self._setup_progress_monitoring()

        try:
            # Run the worker pool
            run_worker_pool(
                num_workers=num_workers,
                src_db_path=self.temp_paths['src_db'],
                work_tracker_path=self.temp_paths['work_tracker'],
                output_dir=self.temp_paths['output_dir'],
                filter_config=self.filter_config,
                worker_config=self._create_worker_config(),
                counters=progress_reporter['counters'] if progress_reporter else None,
            )

            print("  ", "─" * 30, " final ", "─" * 30)

            # Debug output file information
            # self._debug_worker_outputs()

        finally:
            self._cleanup_progress_monitoring(progress_reporter)

    def _setup_progress_monitoring(self) -> Optional[dict]:
        """Set up progress monitoring if enabled."""
        if self.pipeline_config.progress_every_s <= 0:
            return None

        ctx = mp.get_context("spawn")
        counters = create_counters(ctx)
        start_time = time.perf_counter()
        num_workers = getattr(self.pipeline_config, 'readers', 16)

        print_phase_banner()
        stop_event = ctx.Event()

        reporter_process = ctx.Process(
            target=run_progress_reporter,
            args=(counters, start_time, self.pipeline_config.progress_every_s,
                  stop_event, num_workers),
            daemon=True,
            name="ngf:reporter"
        )
        reporter_process.start()

        return {
            'counters': counters,
            'process': reporter_process,
            'stop_event': stop_event
        }

    def _cleanup_progress_monitoring(self, progress_reporter: Optional[dict]) -> None:
        """Clean up progress monitoring resources."""
        if not progress_reporter:
            return

        try:
            progress_reporter['stop_event'].set()
            progress_reporter['process'].join(timeout=2)
        except Exception:
            pass  # Best effort cleanup

    def _debug_worker_outputs(self) -> None:
        """Print debugging information about worker output files."""
        print("\nDEBUGGING: Checking worker output files...")

        output_files = sorted(self.temp_paths['output_dir'].glob("*.db"))
        total_keys_found = 0

        for i, db_file in enumerate(output_files[:10]):  # Check first 10 files
            try:
                with open_db(db_file, mode="ro") as db:
                    count_str = db.get_property("rocksdb.estimate-num-keys")
                    count = int(count_str) if count_str else 0
                    total_keys_found += count
                    print(f"  {db_file.name}: ~{count:,} keys")
            except Exception as e:
                print(f"  ERROR reading {db_file.name}: {e}")

        print(f"  Total keys in first 10 files: {total_keys_found:,}")
        print(f"  Total output files: {len(output_files)}")

    def _merge_results(self) -> None:
        """Merge worker outputs into final database."""
        print(f"\nPhase 3: Merging worker outputs into final database...")
        print("═" * 100)

        output_files = sorted(self.temp_paths['output_dir'].glob("*.db"))
        print(f"  Found {len(output_files)} worker output files")

        if not output_files:
            raise RuntimeError("No worker output files found")

        # Configure merge parameters
        ingest_batch_bytes = getattr(self.pipeline_config, 'ingest_batch_bytes', 64 * 1024 * 1024)
        ingest_batch_items = getattr(self.pipeline_config, 'ingest_batch_items', 100_000)

        print(f"  Batch size: {ingest_batch_bytes // (1024 * 1024)}MB, {ingest_batch_items:,} items")

        # Perform the merge
        self.total_items, self.total_bytes = ingest_shards_streaming(
            dst_db_path=self.temp_paths['dst_db'],
            shards_root=self.temp_paths['output_dir'],
            read_profile=getattr(self.pipeline_config, 'ingest_read_profile', 'read'),
            write_profile=getattr(self.pipeline_config, 'ingest_write_profile', 'bulk_write:packed24'),
            batch_bytes=ingest_batch_bytes,
            batch_items=ingest_batch_items,
            disable_wal=getattr(self.pipeline_config, 'ingest_disable_wal', True),
            diag_every_batches=25,
            diag_every_seconds=3.0
        )

    def _validate_final_result(self) -> None:
        """Validate the final merged database."""
        if not self.temp_paths['dst_db'].exists():
            raise RuntimeError("Final database was not created")

        try:
            with open_db(self.temp_paths['dst_db'], mode="ro") as result_db:
                key_count = result_db.get_property("rocksdb.estimate-num-keys")
        except Exception as e:
            print(f"Could not validate final database: {e}")

    def _generate_output_whitelist(self) -> None:
        """Generate whitelist from final merged database if requested."""
        output_whitelist_path = getattr(self.pipeline_config, "output_whitelist_path", None)
        if not output_whitelist_path:
            return

        output_whitelist_path = Path(output_whitelist_path)
        output_top_n = getattr(self.pipeline_config, "output_whitelist_top_n", None)

        print(f"\nPhase 4: Generating output whitelist...")
        print("═" * 100)
        print(f"  Output path: {output_whitelist_path}")

        if output_top_n:
            print(f"  Extracting top {output_top_n:,} tokens")
        else:
            print(f"  Extracting all tokens")

        start_time = time.perf_counter()
        final_path = write_whitelist(
            db_or_path=self.temp_paths['dst_db'],
            dest=output_whitelist_path,
            top=output_top_n,
            decode=True,
            sep="\t"
        )
        elapsed = time.perf_counter() - start_time

        # Count lines to report how many tokens were written
        try:
            with final_path.open('r', encoding='utf-8') as f:
                token_count = sum(1 for _ in f)
            print(f"  Generated whitelist with {token_count:,} tokens in {elapsed:.1f}s")
        except Exception:
            print(f"  Whitelist generated in {elapsed:.1f}s")

    def _print_done_banner(self) -> None:
        """ Print a banner at the end of the pipeline. """
        db_path_str = str(self.temp_paths['dst_db'])
        if len(db_path_str) > 75:
            db_path_str = "..." + db_path_str[-60:]
        if self.output_whitelist_path:
            whitelist_str = str(self.output_whitelist_path)
        if len(whitelist_str) > 75:
            whitelist_str = "..." + whitelist_str[-60:]

        print("\n╭" + "─" * 85 + "╮")
        message1 = f"PROCESSING COMPLETE: Final DB contains {self.total_items:,} items, {self.total_bytes / 1_000_000:,.1f} MB"
        message2 = "DB Loc: "
        message3 = "Whitelist Loc: "
        print(f"│ {message1:<83} │", flush=True)
        print(f"│ {message2:<83}: {db_path_str} │", flush=True)
        if self.output_whitelist_path:
            print(f"│ {message3:<83}: {whitelist_str} │")
        print("╰" + "─" * 85 + "╯/n")

    def _create_worker_config(self) -> WorkerConfig:
        """Create worker configuration from pipeline config."""
        return WorkerConfig(
            buffer_size=getattr(self.pipeline_config, 'max_items_per_bucket', 25_000),
            buffer_bytes=getattr(self.pipeline_config, 'max_bytes_per_bucket', 16 * 1024 * 1024),
            profile=getattr(self.pipeline_config, 'writer_profile', 'bulk_write:packed24'),
            disable_wal=getattr(self.pipeline_config, 'writer_disable_wal', True),
            disable_compaction=True,
        )


def build_processed_db(pipeline_config: PipelineConfig, filter_config: FilterConfig) -> None:
    """
    Main entry point for the simplified ngram filtering pipeline.

    Args:
        pipeline_config: Configuration for pipeline execution
        filter_config: Configuration for ngram filtering
    """
    orchestrator = PipelineOrchestrator(pipeline_config, filter_config)
    orchestrator.run()