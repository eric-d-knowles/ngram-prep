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
from ngram_filter.partitioning import create_intelligent_work_units
from .work_tracker import WorkTracker
from .worker import WorkerConfig, run_worker_pool
from .ingest import ingest_shards_streaming
from .progress import (
    create_counters,
    print_phase_banner,
    run_progress_reporter,
)
from common_db.api import open_db
from .whitelist import load_whitelist, write_whitelist

__all__ = ["build_processed_db", "PipelineOrchestrator"]

# Display constants
LINE_WIDTH = 100


def _truncate_path_to_fit(
    path: Path | str,
    prefix: str,
    total_width: int = LINE_WIDTH,
) -> str:
    """
    Truncate path to fit within total_width including prefix.

    The total line length (prefix + path) will not exceed total_width.
    Longer prefixes automatically get less space for the path.

    Args:
        path: Path to display
        prefix: The label/prefix before the path
        total_width: Total character width for the entire line

    Returns:
        Truncated path that fits within (total_width - len(prefix))

    Examples:
        >>> _truncate_path_to_fit("/long/path", "Short: ", 50)
        '/long/path'  # Fits within 50 - 7 = 43 chars
        >>> _truncate_path_to_fit("/long/path", "Very long prefix: ", 50)
        '...path'  # Truncated to fit within 50 - 18 = 32 chars
    """
    path_str = str(path)
    max_path_length = total_width - len(prefix)

    if len(path_str) <= max_path_length:
        return path_str

    # Need at least 4 chars for "..."
    if max_path_length < 4:
        return "..."

    return "..." + path_str[-(max_path_length - 3):]


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
        self.temp_paths = self._initialize_paths()

        self.total_items = 0
        self.total_bytes = 0
        self.output_whitelist_path = None

    def _initialize_paths(self) -> dict[str, Path]:
        """
        Initialize and prepare all required paths.

        Returns:
            Dictionary mapping path names to resolved Path objects
        """
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
        enable_ingest = getattr(self.pipeline_config, "enable_ingest", None)

        self._print_pipeline_header()
        self._prepare_directories()
        self._prepare_vocabulary_index()
        self._prepare_whitelist()

        # Execute pipeline phases
        self._create_work_units()
        self._process_work_units()

        if enable_ingest:
            self._merge_results()
            self._validate_final_result()
            self._generate_output_whitelist()
            self._print_done_banner()

    def _print_pipeline_header(self) -> None:
        """Print pipeline configuration summary."""
        print("N-GRAM FILTER PIPELINE")
        print("━" * 100)

        mode = getattr(self.pipeline_config, 'mode', 'resume')
        enable_ingest = getattr(
            self.pipeline_config,
            "enable_ingest",
            None,
        )
        enable_compact = getattr(
            self.pipeline_config,
            "enable_compact",
            None,
        )

        num_workers = getattr(self.pipeline_config, 'readers', 8)
        units_multiplier = getattr(
            self.pipeline_config,
            'work_units_per_reader',
            8,
        )
        num_work_units = num_workers * units_multiplier

        worker_config = self._create_worker_config()

        print("\nConfiguration:")
        print("═" * 100)
        print("\033[4mPipeline\033[0m")
        print(f"Run mode: {mode}")
        print(f"Ingest after filtering: {enable_ingest}")
        print(f"Compact after ingesting: {enable_compact}")
        print("  ")
        print("\033[4mWorkers\033[0m")
        print(f"Num Workers: {num_workers}")
        print(f"Work units: {num_work_units}")

        read_profile = self.pipeline_config.writer_read_profile
        write_profile = self.pipeline_config.writer_write_profile
        print(f"Profiles: read={read_profile}, write={write_profile}")

        buffer_mb = worker_config.buffer_bytes // (1024 * 1024)
        print(
            f"Buffer: {worker_config.buffer_size:,} items, "
            f"{buffer_mb}MB"
        )
        print("  ")
        print("\033[4mFiles\033[0m")

        prefix = "Source: "
        src_path = _truncate_path_to_fit(self.temp_paths['src_db'], prefix)
        print(f"{prefix}{src_path}")

        prefix = "Destination: "
        dst_path = _truncate_path_to_fit(self.temp_paths['dst_db'], prefix)
        print(f"{prefix}{dst_path}")

        # Input whitelist info
        self._print_input_whitelist_info()

        # Output whitelist info
        self.output_whitelist_path = getattr(
            self.pipeline_config,
            "output_whitelist_path",
            None,
        )
        if self.output_whitelist_path:
            output_top_n = getattr(
                self.pipeline_config,
                "output_whitelist_top_n",
                None,
            )
            prefix = "Output whitelist: "
            suffix = f" (top {output_top_n:,} keys)"
            whitelist_path = _truncate_path_to_fit(
                self.output_whitelist_path,
                prefix + suffix,
            )
            print(f"{prefix}{whitelist_path}{suffix}")
        else:
            print("Output whitelist: None")

    def _print_input_whitelist_info(self) -> None:
        """Print input whitelist configuration."""
        whitelist_path = getattr(
            self.filter_config,
            "whitelist_path",
            None,
        )
        if whitelist_path:
            prefix = "Input whitelist: "
            path = _truncate_path_to_fit(whitelist_path, prefix)
            print(f"{prefix}{path}")

            min_count = getattr(
                self.filter_config,
                "whitelist_min_count",
                1,
            )
            top_n = getattr(self.filter_config, "whitelist_top_n", None)
            if top_n:
                print(f"  Top {top_n:,} tokens (min count: {min_count})")
            else:
                print(f"  All tokens (min count: {min_count})")
        else:
            print("Input whitelist: None")

    def _prepare_directories(self) -> None:
        """Clean and create necessary directories."""
        # Always clean destination DB
        if self.temp_paths['dst_db'].exists():
            shutil.rmtree(self.temp_paths['dst_db'])
        self.temp_paths['dst_db'].parent.mkdir(parents=True, exist_ok=True)

        mode = getattr(self.pipeline_config, 'mode', 'resume')

        # Clean temp directory for restart or reprocess modes
        if (
            mode in ('restart', 'reprocess')
            and self.temp_paths['tmp_dir'].exists()
        ):
            shutil.rmtree(self.temp_paths['tmp_dir'])

        # Create temp directories
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
        if self._vocabulary_index_needs_rebuild(
            vocab_path,
            idx_file,
            lex_file,
        ):
            print("Building vocabulary index...")
            build_vocab_index(vocab_path, idx_prefix)

        # Update filter config to use the index
        self.filter_config = replace(
            self.filter_config,
            vocab_path=idx_prefix,
        )

    def _vocabulary_index_needs_rebuild(
        self,
        vocab_path: Path,
        idx_file: Path,
        lex_file: Path,
    ) -> bool:
        """
        Check if vocabulary index needs to be rebuilt.

        Args:
            vocab_path: Path to vocabulary file
            idx_file: Path to index file
            lex_file: Path to lexicon file

        Returns:
            True if rebuild needed, False otherwise
        """
        if not idx_file.exists() or not lex_file.exists():
            return True

        # Check if vocab file is newer than index files
        vocab_mtime = vocab_path.stat().st_mtime
        idx_mtime = max(
            idx_file.stat().st_mtime,
            lex_file.stat().st_mtime,
        )
        return vocab_mtime > idx_mtime

    def _prepare_whitelist(self) -> None:
        """Load whitelist if specified in filter config."""
        whitelist_path = getattr(
            self.filter_config,
            "whitelist_path",
            None,
        )
        if not whitelist_path:
            return

        whitelist_path = Path(whitelist_path)
        if not whitelist_path.exists():
            raise FileNotFoundError(
                f"Whitelist file not found: {whitelist_path}"
            )

        print("Loading whitelist...")

        # Load whitelist with optional parameters from config
        min_count = getattr(self.filter_config, "whitelist_min_count", 1)
        top_n = getattr(self.filter_config, "whitelist_top_n", None)

        whitelist = load_whitelist(
            whitelist_path=whitelist_path,
            min_count=min_count,
            top_n=top_n,
        )

        print(f"Loaded {len(whitelist):,} tokens")

        # Store the loaded whitelist in filter_config for workers
        self.filter_config = replace(
            self.filter_config,
            whitelist=whitelist,
        )

    def _create_work_units(self) -> None:
        """Create or resume work units for processing."""
        print("\nPhase 1: Creating work units...")
        print("═" * 100)

        work_tracker = WorkTracker(self.temp_paths['work_tracker'])
        num_workers = getattr(self.pipeline_config, 'readers', 8)
        units_multiplier = getattr(
            self.pipeline_config,
            'work_units_per_reader',
            64,
        )
        num_work_units = num_workers * units_multiplier

        mode = getattr(self.pipeline_config, 'mode', 'resume')

        if mode == 'restart':
            print("Clean restart - resampling and creating new work units\n")
            work_tracker.clear_all_work_units()
            self._create_new_work_units(work_tracker, num_work_units)

        elif mode == 'reprocess':
            print(
                "Reprocess - loading cached work units and resetting status"
            )
            work_tracker.clear_all_work_units()
            self._create_new_work_units(work_tracker, num_work_units)

        elif mode == 'resume':
            progress = work_tracker.get_progress()
            if progress.total == 0:
                print("No existing work units - loading from cache")
                self._create_new_work_units(work_tracker, num_work_units)
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
        num_work_units: int,
    ) -> None:
        """
        Create work units from cache or by resampling.

        Args:
            work_tracker: WorkTracker instance
            num_work_units: Number of work units to create
        """
        mode = getattr(self.pipeline_config, 'mode', 'resume')
        sample_rate = getattr(
            self.pipeline_config,
            'partitioning_sample_rate',
            0.001,
        )
        prefix_length = getattr(self.pipeline_config, 'prefix_length', 2)
        force_cache_use = getattr(
            self.pipeline_config,
            'force_cache_use',
            False,
        )

        work_units = create_intelligent_work_units(
            self.temp_paths['src_db'],
            num_work_units,
            sample_rate=sample_rate,
            prefix_length=prefix_length,
            use_cache=(mode != 'restart'),
            force_resample=(mode == 'restart'),
            force_cache_use=force_cache_use
        )

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
        work_tracker = WorkTracker(self.temp_paths['work_tracker'])
        progress = work_tracker.get_progress()
        num_workers = getattr(self.pipeline_config, 'readers', 16)

        print(
            f"\nPhase 2: Processing {progress.pending} work units "
            f"with {num_workers} workers..."
        )
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
                pipeline_config=self.pipeline_config,
                worker_config=self._create_worker_config(),
                counters=(
                    progress_reporter['counters']
                    if progress_reporter
                    else None
                ),
            )

            print(f"   {'─' * 36} final {'─' * 36}")

        finally:
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
        num_workers = getattr(self.pipeline_config, 'readers', 16)

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

    def _merge_results(self) -> None:
        """Merge worker outputs into final database."""
        print("\nPhase 3: Merging worker outputs into final database...")
        print("═" * 100)

        output_files = sorted(self.temp_paths['output_dir'].glob("*.db"))
        print(f"Found {len(output_files)} worker output files")

        if not output_files:
            raise RuntimeError("No worker output files found")

        # Configure merge parameters
        ingest_batch_bytes = getattr(
            self.pipeline_config,
            'ingest_batch_bytes',
            64 * 1024 * 1024,
        )
        ingest_batch_items = getattr(
            self.pipeline_config,
            'ingest_batch_items',
            100_000,
        )
        num_ingestors = getattr(self.pipeline_config, 'ingestors', 8)
        delete_shards = getattr(
            self.pipeline_config,
            'delete_after_ingest',
            False,
        )
        enable_compact = getattr(
            self.pipeline_config,
            'enable_compact',
            True,
        )

        batch_mb = ingest_batch_bytes // (1024 * 1024)
        print(
            f"Ingesting shards with batch size {batch_mb}MB / "
            f"{ingest_batch_items:,} items"
        )
        if delete_shards:
            print("Deleting shards after successful ingestion")
        else:
            print("Retaining shards after successful ingestion")

        # Perform the merge
        self.total_items, self.total_bytes = ingest_shards_streaming(
            dst_db_path=self.temp_paths['dst_db'],
            shards_root=self.temp_paths['output_dir'],
            read_profile=getattr(
                self.pipeline_config,
                'ingest_read_profile',
                'read:packed24',
            ),
            write_profile=getattr(
                self.pipeline_config,
                'ingest_write_profile',
                'write:packed24',
            ),
            batch_bytes=ingest_batch_bytes,
            batch_items=ingest_batch_items,
            disable_wal=getattr(
                self.pipeline_config,
                'ingest_disable_wal',
                True,
            ),
            diag_every_batches=25,
            diag_every_seconds=3.0,
            num_readers=num_ingestors,
            delete_after_ingest=delete_shards,
            enable_compact=enable_compact
        )

    def _validate_final_result(self) -> None:
        """Validate the final merged database."""
        if not self.temp_paths['dst_db'].exists():
            raise RuntimeError("Final database was not created")

        try:
            with open_db(self.temp_paths['dst_db'], mode="ro") as result_db:
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

        print("\nPhase 4: Generating output whitelist...")
        print("═" * 100)

        prefix = "  Output path: "
        path = _truncate_path_to_fit(output_whitelist_path, prefix)
        print(f"{prefix}{path}")

        if output_top_n:
            print(f"  Extracting top {output_top_n:,} tokens")
        else:
            print("  Extracting all tokens")

        start_time = time.perf_counter()
        final_path = write_whitelist(
            db_or_path=self.temp_paths['dst_db'],
            dest=output_whitelist_path,
            top=output_top_n,
            decode=True,
            sep="\t"
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

    def _print_done_banner(self) -> None:
        """Print completion banner."""
        prefix = "│ DB: "
        db_path_str = _truncate_path_to_fit(
            self.temp_paths['dst_db'],
            prefix,
            total_width=87,  # Width of box minus borders
        )

        whitelist_str = ""
        if self.output_whitelist_path:
            prefix = "│ Whitelist: "
            whitelist_str = _truncate_path_to_fit(
                self.output_whitelist_path,
                prefix,
                total_width=87,
            )

        print("\n╭" + "─" * 87 + "╮")
        total_mb = self.total_bytes / 1_000_000
        message1 = (
            f"PROCESSING COMPLETE: Final DB contains "
            f"{self.total_items:,} items, {total_mb:,.1f} MB"
        )
        print(f"│ {message1:<83}   │")
        print(f"│ DB: {db_path_str:<79} │")
        if whitelist_str:
            print(f"│ Whitelist: {whitelist_str:<72} │")
        print("╰" + "─" * 87 + "╯\n")

    def _create_worker_config(self) -> WorkerConfig:
        """
        Create worker configuration from pipeline config.

        Returns:
            Configured WorkerConfig instance
        """
        return WorkerConfig(
            buffer_size=getattr(
                self.pipeline_config,
                'max_items_per_bucket',
                25_000,
            ),
            buffer_bytes=getattr(
                self.pipeline_config,
                'max_bytes_per_bucket',
                16 * 1024 * 1024,
            ),
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
