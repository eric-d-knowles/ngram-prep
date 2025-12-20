"""Core filtering pipeline for Davies corpus."""
from __future__ import annotations

import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, Set, Any, List
from collections import defaultdict

from tqdm import tqdm

from ngramprep.common_db.api import open_db
from ngramprep.ngram_pivot.encoding import decode_year_ngram_key, encode_year_ngram_key, encode_year_stats, decode_year_stats
from ngramprep.utilities.display import format_banner, format_bytes

from ..davies_acquire.encoding import decode_occurrence_count
from .config import FilterConfig, PipelineConfig
from .filters.processor_factory import build_processor
from .whitelist import write_whitelist, load_whitelist

__all__ = ["filter_davies_corpus"]

# Try to import setproctitle (optional dependency)
try:
    import setproctitle as _setproctitle
except ImportError:
    _setproctitle = None


def _perform_compaction(db, db_path: Path, initial_size: Optional[int] = None) -> None:
    """
    Perform full compaction on the database using compact_all().

    Args:
        db: Open RocksDB handle
        db_path: Path to database (for logging)
        initial_size: Pre-compaction size in bytes (if already known)
    """
    print()
    print(format_banner("Post-Filter Compaction"))

    # Get or display initial size
    if initial_size is not None and initial_size > 0:
        print(f"Initial DB size:         {format_bytes(initial_size)}")
    else:
        # Try to get size if not provided
        try:
            initial_size_str = db.get_property(b"rocksdb.total-sst-files-size")
            if initial_size_str:
                initial_size = int(initial_size_str)
                if initial_size > 0:
                    print(f"Initial DB size:         {format_bytes(initial_size)}")
        except Exception:
            initial_size = None

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
        print(f"Compaction failed: {e}")
        print("Database is still usable, but may not be optimally compacted.")

    print()


def process_batch(
    batch_id: int,
    batch: List[Tuple[bytes, bytes]],
    filter_config_dict: Dict[str, Any],
) -> Tuple[int, int, Dict[bytes, bytes]]:
    """
    Process a single batch of sentences (runs in worker process via ProcessPoolExecutor).

    Args:
        batch_id: Batch identifier for process naming
        batch: List of (key_bytes, value_bytes) tuples in year-only format
        filter_config_dict: Filter configuration as dict

    Returns:
        Tuple of (sentences_processed, sentences_rejected, filtered_data)
    """
    # Set process title if available
    if _setproctitle is not None:
        try:
            _setproctitle.setproctitle(f"davf:worker[{batch_id:03d}]")
        except Exception:
            pass

    # Reconstruct FilterConfig from dict in worker process
    # If lemmatization is enabled but no lemmatizer provided, create one
    if filter_config_dict.get('apply_lemmatization', False) and filter_config_dict.get('lemma_gen') is None:
        from ngramprep.ngram_filter.lemmatizer import CachedSpacyLemmatizer
        filter_config_dict = dict(filter_config_dict)  # Make mutable copy
        filter_config_dict['lemma_gen'] = CachedSpacyLemmatizer()

    filter_config = FilterConfig(**filter_config_dict)

    # Build processor (creates lemmatizer if needed)
    processor = build_processor(filter_config)

    sentences_processed = 0
    sentences_rejected = 0
    filtered_data: List[Tuple[bytes, bytes]] = []  # List of (key, value) tuples, let RocksDB merge

    for key_bytes, value_bytes in batch:
        sentences_processed += 1

        # Decode year-only format key: [year:4][sentence]
        try:
            year, sentence_bytes = decode_year_ngram_key(key_bytes)
        except Exception as e:
            # Skip keys that can't be decoded
            sentences_rejected += 1
            continue

        # Apply filter
        filtered_sentence = processor(sentence_bytes)

        if filtered_sentence is None:
            sentences_rejected += 1
        else:
            # Create new key with filtered sentence
            filtered_tokens = filtered_sentence.decode('utf-8').split()

            # Always output year-only format (no genre prefix)
            # This is required for training compatibility
            filtered_sentence_bytes = ' '.join(filtered_tokens).encode('utf-8')
            new_key = encode_year_ngram_key(year, filtered_sentence_bytes)

            # Pass through the value unchanged - it may be a merged value
            # This preserves occurrence counts from previous phases
            new_value = value_bytes

            # Just append - let RocksDB merge operator handle duplicates
            filtered_data.append((new_key, new_value))

    return sentences_processed, sentences_rejected, filtered_data


def _run_filtering_workers(
    batches: List[List[Tuple[bytes, bytes]]],
    dst_db,
    filter_config_dict: Dict[str, Any],
    workers: int,
) -> Tuple[int, int, int]:
    """
    Run worker processes to filter batches and write results to database.

    Uses ProcessPoolExecutor (like Davies acquisition) to avoid manual queue management.

    Args:
        batches: List of batches to process (year-only format keys)
        dst_db: Open destination database handle
        filter_config_dict: Filter configuration as dict
        workers: Number of worker processes

    Returns:
        Tuple of (sentences_read, sentences_rejected, unique_sentences_written)
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    sentences_read = 0
    sentences_rejected = 0
    writes_accumulated = 0

    # Accumulate writes and let RocksDB merge operator handle duplicates
    accumulated_writes: List[Tuple[bytes, bytes]] = []
    WRITE_BATCH_SIZE = 50_000

    # Process batches with ProcessPoolExecutor (like Davies acquisition)
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        with tqdm(
            total=len(batches),
            desc="Batches Processed",
            unit="batches",
            ncols=100,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ) as pbar:
            # Submit all batches for processing
            future_to_batch = {
                executor.submit(process_batch, batch_id, batch, filter_config_dict): batch_id
                for batch_id, batch in enumerate(batches)
            }

            # Process completed futures as they finish
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_processed, batch_rejected, filtered_data = future.result()

                    sentences_read += batch_processed
                    sentences_rejected += batch_rejected

                    # Accumulate all writes - let RocksDB merge operator handle duplicates
                    accumulated_writes.extend(filtered_data)
                    writes_accumulated += len(filtered_data)

                    # Flush accumulated writes periodically using write_batch with merge
                    if len(accumulated_writes) >= WRITE_BATCH_SIZE:
                        with dst_db.write_batch(disable_wal=True, sync=False) as wb:
                            for k, v in accumulated_writes:
                                wb.merge(k, v)
                        # Context manager auto-commits, merge operator handles duplicates
                        accumulated_writes.clear()

                except Exception as e:
                    print(f"Error processing batch {batch_id}: {e}")
                    raise
                finally:
                    pbar.update(1)
    finally:
        # Explicitly shutdown executor without waiting indefinitely for process cleanup
        executor.shutdown(wait=False)
        # Force cleanup of any remaining references
        del executor

    # Flush any remaining writes
    if accumulated_writes:
        with dst_db.write_batch(disable_wal=True, sync=False) as wb:
            for k, v in accumulated_writes:
                wb.merge(k, v)
        # Context manager auto-commits, merge operator handles duplicates

    return sentences_read, sentences_rejected, writes_accumulated


def filter_davies_corpus(
    db_path_stub: str | Path,
    genre_focus: Optional[List[str]] = None,
    filter_config: Optional[FilterConfig] = None,
    # Filter configuration parameters (used if filter_config not provided)
    stop_set: Optional[Set[str]] = None,
    lemma_gen: Any = None,
    lowercase: Optional[bool] = None,
    alpha_only: Optional[bool] = None,
    filter_short: Optional[bool] = None,
    filter_stops: Optional[bool] = None,
    apply_lemmatization: Optional[bool] = None,
    min_len: Optional[int] = None,
    whitelist: Optional[Set[bytes]] = None,
    always_include: Optional[Set[str]] = None,
    # Processing parameters
    workers: Optional[int] = None,
    batch_size: int = 50_000,
    compact_after: bool = False,
    # Whitelist creation and application parameters
    create_whitelist: bool = False,
    whitelist_size: int = 10_000,
    whitelist_year_range: Optional[Tuple[int, int]] = None,
    whitelist_spell_check: bool = True,
    whitelist_workers: Optional[int] = None,
    whitelist_batch_size: int = 50_000,
    apply_whitelist: bool = False,
) -> None:
    """
    Filter Davies corpus database with parallel processing.

    Reads sentences from source database (year-only format), applies filters in parallel,
    writes to filtered database.

    Directory structure:
        db_path_stub/
            {corpus_name}              <- Source DB (if genre_focus=None)
            {corpus_name}_fic          <- Source DB (if genre_focus=["fic"])
            {corpus_name}_filtered     <- Output DB (created by this function)
            {corpus_name}_fic_filtered <- Output DB (if genre_focus=["fic"])
            {corpus_name}_whitelist.txt     <- Whitelist file
            {corpus_name}_fic_whitelist.txt <- Whitelist file (if genre_focus=["fic"])

    NOTE: The genre_focus parameter should match what was used during ingestion with
    ingest_davies_corpus(). It's used to locate the correct source database.

    Args:
        db_path_stub: Path to corpus directory (e.g., "/path/to/COHA").
        genre_focus: Optional list of genres (e.g., ["fic"]). Should match the genre_focus
                    used during ingestion. If None, looks for plain corpus database (e.g., "COHA").
                    If specified, looks for genre-suffixed database (e.g., "COHA_fic").
        filter_config: Filtering configuration (uses defaults if not provided)
        stop_set: Set of stopwords to filter (used if filter_config not provided)
        lemma_gen: Lemmatizer instance (used if filter_config not provided)
        lowercase: Apply lowercasing (used if filter_config not provided)
        alpha_only: Filter non-alphabetic tokens (used if filter_config not provided)
        filter_short: Filter short tokens (used if filter_config not provided)
        filter_stops: Filter stopwords (used if filter_config not provided)
        apply_lemmatization: Apply lemmatization (used if filter_config not provided)
        min_len: Minimum token length (used if filter_config not provided)
        whitelist: Set of allowed tokens (bytes); tokens not in whitelist become <UNK>
        always_include: Set of tokens (strings) to always preserve in whitelist mode,
                       regardless of whether they're in the whitelist (e.g., {"working-class", "nuclear"})
        workers: Number of parallel workers (default: cpu_count - 1)
        batch_size: Number of sentences per batch for workers
        compact_after: If True, perform full compaction after filtering
        create_whitelist: If True, build whitelist from filtered database
        whitelist_path: Path to save whitelist file (required if create_whitelist=True)
        whitelist_size: Number of top tokens to include in whitelist
        whitelist_year_range: Optional (start_year, end_year) range for whitelist creation
        whitelist_spell_check: If True, filter out misspelled words from whitelist
        whitelist_workers: Number of parallel workers for whitelist building (default: same as workers)
        whitelist_batch_size: Batch size for whitelist building (default: 50,000)
        apply_whitelist: If True, apply whitelist after creation (requires create_whitelist=True)

    Workflow:
        1. Phase 1: Apply initial filters (lowercase, lemmatize, alpha-only, stops, short words)
        2. Phase 2 (if create_whitelist=True): Build whitelist from filtered database
        3. Phase 3 (if apply_whitelist=True): Replace non-whitelist tokens with <UNK>
    """
    # Set main process title if available
    if _setproctitle is not None:
        try:
            _setproctitle.setproctitle("davf:main")
        except Exception:
            pass

    start_time = datetime.now()

    # Validate whitelist parameters
    if apply_whitelist and not create_whitelist:
        raise ValueError("apply_whitelist=True requires create_whitelist=True")

    # Parse db_path_stub and auto-detect source database
    db_path_stub_obj = Path(db_path_stub)
    corpus_name = db_path_stub_obj.name  # e.g., "COHA" from "/path/to/COHA"

    if not db_path_stub_obj.exists():
        raise ValueError(f"Corpus directory does not exist: {db_path_stub_obj}")

    # Determine source database name based on genre_focus
    if genre_focus is not None:
        # Build database name with genre suffix (matching ingestion)
        genre_suffix = "+".join(sorted(genre_focus))
        src_db_name = f"{corpus_name}_{genre_suffix}"
    else:
        # Plain corpus name (no genre filtering)
        src_db_name = corpus_name

    src_db_path = db_path_stub_obj / src_db_name

    if not src_db_path.exists():
        raise ValueError(
            f"Source database not found: {src_db_path}. "
            f"Make sure you ran ingest_davies_corpus() with genre_focus={genre_focus} first."
        )

    # Destination database name is source + "_filtered"
    dst_db_path = db_path_stub_obj / f"{src_db_name}_filtered"

    # Whitelist path is auto-derived if create_whitelist=True
    if create_whitelist:
        whitelist_path = db_path_stub_obj / f"{src_db_name}_whitelist.txt"
    else:
        whitelist_path = None

    # Handle existing destination database - always remove for fresh start
    if dst_db_path.exists():
        from ngramprep.ngram_acquire.utils.cleanup import safe_db_cleanup
        if not safe_db_cleanup(dst_db_path):
            raise RuntimeError(
                f"Failed to remove existing database at {dst_db_path}. "
                "Close open handles or remove it manually."
            )

    # Ensure parent directory exists
    dst_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine worker count
    if workers is None:
        cpu_count = os.cpu_count() or 4
        workers = max(1, cpu_count - 1)

    # Convert always_include to bytes if provided
    always_include_bytes = None
    if always_include is not None:
        always_include_bytes = {token.encode('utf-8') for token in always_include}

    # Construct FilterConfig if not provided
    if filter_config is None:
        # If filter parameters provided, use them; otherwise use defaults
        if any(param is not None for param in [stop_set, lemma_gen, lowercase, alpha_only,
                                                 filter_short, filter_stops, apply_lemmatization, min_len, whitelist, always_include_bytes]):
            kwargs = {}
            if stop_set is not None:
                kwargs['stop_set'] = stop_set
            if lemma_gen is not None:
                kwargs['lemma_gen'] = lemma_gen
            if lowercase is not None:
                kwargs['lowercase'] = lowercase
            if alpha_only is not None:
                kwargs['alpha_only'] = alpha_only
            if filter_short is not None:
                kwargs['filter_short'] = filter_short
            if filter_stops is not None:
                kwargs['filter_stops'] = filter_stops
            if apply_lemmatization is not None:
                kwargs['apply_lemmatization'] = apply_lemmatization
            if min_len is not None:
                kwargs['min_len'] = min_len
            if whitelist is not None:
                kwargs['whitelist'] = whitelist
            if always_include_bytes is not None:
                kwargs['always_include'] = always_include_bytes
            filter_config = FilterConfig(**kwargs)
        else:
            filter_config = FilterConfig()

    # Print header
    print(format_banner(f"{corpus_name} CORPUS FILTERING", style="━"))
    print(f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}")
    print()
    print(format_banner("Configuration"))
    print(f"Source DB:            {src_db_path}")
    print(f"Destination DB:       {dst_db_path}")
    print(f"Lowercase:            {filter_config.lowercase}")
    print(f"Alpha only:           {filter_config.alpha_only}")
    print(f"Filter short:         {filter_config.filter_short} (min_len={filter_config.min_len})")
    print(f"Filter stops:         {filter_config.filter_stops}")
    print(f"Apply lemmas:         {filter_config.apply_lemmatization}")
    print(f"Workers:              {workers}")
    print(f"Batch size:           {batch_size:,}")
    print()
    print(format_banner("Processing Sentences"))
    sys.stdout.flush()

    # First pass: collect all keys and values into batches
    batches: List[List[Tuple[bytes, bytes]]] = []
    current_batch: List[Tuple[bytes, bytes]] = []

    with open_db(src_db_path, mode="r", profile="read:packed24") as src_db:
        iterator = src_db.iterator()
        iterator.seek(b"")

        while iterator.valid():
            key_bytes = iterator.key()
            value_bytes = iterator.value()
            current_batch.append((key_bytes, value_bytes))

            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []

            iterator.next()

        # Add remaining batch
        if current_batch:
            batches.append(current_batch)

        # Explicitly cleanup iterator
        del iterator

    total_sentences = sum(len(batch) for batch in batches)

    # Convert FilterConfig to dict for pickling (avoids lemmatizer pickling issues)
    # Workers will reconstruct the config and build their own lemmatizers
    filter_config_dict = {
        'lowercase': filter_config.lowercase,
        'alpha_only': filter_config.alpha_only,
        'filter_short': filter_config.filter_short,
        'filter_stops': filter_config.filter_stops,
        'apply_lemmatization': filter_config.apply_lemmatization,
        'min_len': filter_config.min_len,
        'stop_set': filter_config.stop_set,
        # Don't include lemma_gen - let workers create their own
    }

    # Only include whitelist if it's set (requires recompiled Cython extension)
    if filter_config.whitelist is not None:
        filter_config_dict['whitelist'] = filter_config.whitelist

    # Include always_include if it's set
    if filter_config.always_include is not None:
        filter_config_dict['always_include'] = filter_config.always_include

    # Open destination database and run filtering workers
    with open_db(dst_db_path, mode="w", profile="write:packed24", create_if_missing=True) as dst_db:
        # Run workers in separate function that returns cleanly
        # Input is always year-only format from ingestion
        sentences_read, sentences_rejected, writes_accumulated = _run_filtering_workers(
            batches=batches,
            dst_db=dst_db,
            filter_config_dict=filter_config_dict,
            workers=workers,
        )

        # Context manager will call finalize_bulk() and close() on exit

    # Phase 2: Create whitelist if requested
    whitelist_set: Optional[Set[bytes]] = None
    if create_whitelist:
        print()
        print(format_banner("Building Whitelist"))
        print(f"Whitelist path:          {whitelist_path}")
        print(f"Top N tokens:            {whitelist_size:,}")
        if whitelist_year_range:
            print(f"Year range:              {whitelist_year_range[0]}-{whitelist_year_range[1]}")
        print(f"Spell check:             {whitelist_spell_check}")
        print()
        sys.stdout.flush()

        # Determine whitelist worker count
        wl_workers = whitelist_workers if whitelist_workers is not None else workers

        # Filtered database is always in year-only format
        # So whitelist builder should expect year-only keys
        # Call write_whitelist to create the whitelist file
        write_whitelist(
            db_or_path=dst_db_path,
            dest=whitelist_path,
            top=whitelist_size,
            track_genre=False,  # Filtered DB is always year-only format
            genre_focus=None,  # Not used - genre filtering done at ingestion
            spell_check=whitelist_spell_check,
            alpha_only=filter_config.alpha_only,  # Pass through the filter setting
            year_range=whitelist_year_range,
            always_include=always_include_bytes,
            workers=wl_workers,
            batch_size=whitelist_batch_size,
        )

        sys.stdout.flush()

    # Phase 3: Apply whitelist if requested
    if apply_whitelist and create_whitelist:
        print()
        print(format_banner("Applying Whitelist"))
        print("Loading whitelist into memory...")
        whitelist_set = load_whitelist(whitelist_path)
        print(f"Loaded {len(whitelist_set):,} tokens from whitelist")
        print()
        print(f"Replacing non-whitelist tokens with <UNK>...")
        print()
        sys.stdout.flush()

        # Re-filter the database with whitelist
        # We need to re-read all sentences and apply whitelist filter
        from .filters.processor_factory import build_processor

        # Build a new FilterConfig with the whitelist
        whitelist_filter_config = FilterConfig(
            lowercase=False,  # Already done in phase 1
            alpha_only=False,  # Already done in phase 1
            filter_short=False,  # Already done in phase 1
            filter_stops=False,  # Already done in phase 1
            apply_lemmatization=False,  # Already done in phase 1
            whitelist=whitelist_set,  # Only apply whitelist filter
        )

        # Collect all sentences from filtered database
        whitelist_batches: List[List[Tuple[bytes, bytes]]] = []
        current_batch: List[Tuple[bytes, bytes]] = []

        with open_db(dst_db_path, mode="r", profile="read:packed24") as src_db:
            iterator = src_db.iterator()
            iterator.seek(b"")

            while iterator.valid():
                key_bytes = iterator.key()
                value_bytes = iterator.value()
                current_batch.append((key_bytes, value_bytes))

                if len(current_batch) >= batch_size:
                    whitelist_batches.append(current_batch)
                    current_batch = []

                iterator.next()

            if current_batch:
                whitelist_batches.append(current_batch)

            del iterator

        # Convert FilterConfig to dict for workers
        # IMPORTANT: Add <UNK> to always_include so it doesn't get filtered in whitelist pass
        whitelist_always_include = {b'<UNK>'}
        if always_include_bytes:
            whitelist_always_include |= always_include_bytes

        whitelist_filter_config_dict = {
            'lowercase': False,
            'alpha_only': False,
            'filter_short': False,
            'filter_stops': False,
            'apply_lemmatization': False,
            'min_len': 3,
            'stop_set': set(),
            'whitelist': whitelist_set,
            'always_include': whitelist_always_include,
        }

        # Write to temporary database to avoid overwriting source
        import tempfile
        import shutil
        temp_db_path = Path(tempfile.mkdtemp(prefix=f"{corpus_name}_wl_", dir=dst_db_path.parent))

        try:
            # Apply whitelist filter to temporary database
            # Note: Filtered database is always in year-only format
            with open_db(temp_db_path, mode="w", profile="write:packed24", create_if_missing=True) as temp_db:
                wl_sentences_read, wl_sentences_rejected, wl_writes_accumulated = _run_filtering_workers(
                    batches=whitelist_batches,
                    dst_db=temp_db,
                    filter_config_dict=whitelist_filter_config_dict,
                    workers=workers,
                )

            # Replace original database with whitelist-filtered version
            from ngramprep.ngram_acquire.utils.cleanup import safe_db_cleanup
            if not safe_db_cleanup(dst_db_path):
                raise RuntimeError(f"Failed to remove original database at {dst_db_path}")

            # Move temp database to final location
            shutil.move(str(temp_db_path), str(dst_db_path))

            print(f"\nWhitelist application complete!")
            print(f"Sentences processed:      {wl_sentences_read:,}")
            print(f"Sentences modified:       {wl_sentences_rejected:,}")
            sys.stdout.flush()

        finally:
            # Cleanup temp database if it still exists
            if temp_db_path.exists():
                try:
                    shutil.rmtree(temp_db_path)
                except Exception:
                    pass

        # Cleanup
        del whitelist_batches

    # Optional post-filter compaction (after all phases)
    if compact_after:
        with open_db(dst_db_path, mode="w", profile="write:packed24") as dst_db:
            # Get initial size before flushing
            initial_size = None
            try:
                initial_size_str = dst_db.get_property(b"rocksdb.total-sst-files-size")
                if initial_size_str:
                    initial_size = int(initial_size_str)
            except Exception:
                pass

            # Flush memtables to SST files before compaction
            try:
                dst_db.finalize_bulk()
            except AttributeError:
                pass  # Method not available

            # If we didn't get size before, try again after flush
            if initial_size is None or initial_size == 0:
                try:
                    initial_size_str = dst_db.get_property(b"rocksdb.total-sst-files-size")
                    if initial_size_str:
                        initial_size = int(initial_size_str)
                except Exception:
                    pass

            _perform_compaction(dst_db, dst_db_path, initial_size=initial_size)

    end_time = datetime.now()
    elapsed = end_time - start_time

    print("\nProcessing complete!")
    print()
    print(format_banner("Final Summary"))
    print(f"Sentences read:           {sentences_read:,}")
    print(f"Writes accumulated:       {writes_accumulated:,}")
    print(f"Sentences rejected:       {sentences_rejected:,}")
    retention_pct = (writes_accumulated / sentences_read * 100) if sentences_read > 0 else 0
    print(f"Retention rate:           {retention_pct:.1f}%")
    print(f"Destination DB:           {dst_db_path}")
    print()
    print(f"End Time: {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"Total Runtime: {elapsed}")
    print()
    sys.stdout.flush()

    # Cleanup all references
    del batches

    # Reset process title
    if _setproctitle is not None:
        try:
            _setproctitle.setproctitle("davf:done")
        except Exception:
            pass
