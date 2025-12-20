"""Main entry point for Davies corpus acquisition pipeline."""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from ngramprep.common_db.api import open_db
from ngramprep.utilities.display import format_banner, format_bytes

from .reader import discover_text_files, extract_year_from_filename, read_text_file_with_genre
from .tokenizer import tokenize_sentences

logger = logging.getLogger(__name__)

__all__ = ["ingest_davies_corpus"]

# Try to import setproctitle (optional dependency)
try:
    import setproctitle as _setproctitle
except ImportError:
    _setproctitle = None


def process_single_file(
    zip_path: Path,
    year: int,
    worker_id: int = 0,
    combined_bigrams: Optional[set] = None,
    genre_focus: Optional[List[str]] = None,
) -> Tuple[str, int, int, Dict, Dict[str, int]]:
    """
    Process a single text file: read, tokenize, accumulate sentence counts.

    This function runs in a worker process and returns sentence counts to be merged
    into the main database.

    Args:
        zip_path: Path to zip file
        year: Year for this file
        worker_id: Worker identifier for process naming
        combined_bigrams: Optional set of bigrams to combine with hyphens
                         (e.g., {"working class", "middle class"})
        genre_focus: Optional list of genres to include (e.g., ["fic", "mag"]).
                    If None, include all genres.

    Returns:
        Tuple of (filename, sentence_count, error_count, sentence_data, genre_stats)
        where sentence_data is Dict[(genre, year, sentence_str)] -> count
        and genre_stats is Dict[genre] -> count
    """
    from collections import defaultdict

    # Set process title if available (helps with process monitoring)
    if _setproctitle is not None:
        try:
            _setproctitle.setproctitle(f"dava:worker[{worker_id:03d}]")
        except Exception:
            pass

    sentence_count = 0
    error_count = 0
    sentence_data: Dict = defaultdict(int)
    genre_stats: Dict[str, int] = defaultdict(int)

    try:
        # Read documents from zip file with genre
        for doc_year, text, genre in read_text_file_with_genre(zip_path, year):
            # Skip if genre filtering is enabled and this genre is not in focus
            genre_key = genre if genre is not None else 'unknown'
            if genre_focus is not None and genre_key not in genre_focus:
                continue

            try:
                # Tokenize into sentences (with optional bigram combination)
                for tokens in tokenize_sentences(text, combined_bigrams=combined_bigrams):
                    sentence_str = ' '.join(tokens)

                    # Include genre in key: (genre, year, sentence_str)
                    sentence_data[(genre_key, doc_year, sentence_str)] += 1
                    genre_stats[genre_key] += 1

                    sentence_count += 1
            except Exception as e:
                logger.warning(f"Error tokenizing document in {zip_path.name}: {e}")
                error_count += 1

    except Exception as e:
        logger.error(f"Error processing {zip_path.name}: {e}")
        error_count += 1

    return zip_path.name, sentence_count, error_count, sentence_data, genre_stats


def _perform_compaction(db, db_path: Path) -> None:
    """
    Perform full compaction on the database using compact_all().

    Args:
        db: Open RocksDB handle
        db_path: Path to database (for logging)
    """
    logger.info("Starting post-ingestion compaction")
    print()
    print(format_banner("Post-Ingestion Compaction"))

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


def ingest_davies_corpus(
    db_path_stub: str,
    workers: Optional[int] = None,
    write_batch_size: int = 100_000,
    compact_after: bool = False,
    combined_bigrams: Optional[set] = None,
    genre_focus: Optional[List[str]] = None,
) -> None:
    """
    Main pipeline: read Davies corpus text files and ingest into RocksDB.

    Orchestrates the complete Davies acquisition workflow:
    1. Discovers text files in corpus directory
    2. Opens/creates RocksDB in pivoted format
    3. Reads and tokenizes text files with genre information
    4. Writes sentences to DB (format depends on genre_focus parameter)
    5. Optionally performs post-ingestion compaction

    Directory structure:
        db_path_stub/
            text/               <- Corpus text files (input)
            {corpus_name}       <- Database (output, if genre_focus=None)
            {corpus_name}_fic   <- Database (output, if genre_focus=["fic"])

    Key format behavior:
    - If genre_focus is None: Writes genre-prefixed keys (genre, year, tokens) for archival use
      Database name: {corpus_name} (e.g., "COHA")
    - If genre_focus is specified: Filters to those genres and writes year-only keys (year, tokens)
      Database name: {corpus_name}_{genre1}+{genre2} (e.g., "COHA_fic" or "COHA_fic+mag")

    Args:
        db_path_stub: Path to corpus directory (e.g., "/path/to/COHA"). Should contain a 'text/'
                     subdirectory with corpus files. Corpus name is derived from directory name.
                     Database will be created in this same directory.
        workers: Number of concurrent workers (default: cpu_count - 1)
        write_batch_size: Number of sentences per batch write
        compact_after: If True, perform full compaction after ingestion
        combined_bigrams: Optional set of bigrams to combine with hyphens during tokenization
                         (e.g., {"working class", "middle class"}). Consecutive tokens matching
                         these bigrams will be replaced with hyphenated versions (e.g., "working-class")
        genre_focus: Optional list of genres to ingest (e.g., ["fic", "mag"]). If None, ingest all
                    genres with genre-prefixed keys. If specified, only ingest those genres and use
                    year-only keys for direct training compatibility.
    """
    # Set main process title if available
    if _setproctitle is not None:
        try:
            _setproctitle.setproctitle("dava:main")
        except Exception:
            pass

    logger.info("Starting Davies corpus acquisition pipeline")
    start_time = datetime.now()

    # Parse db_path_stub
    db_path_stub_obj = Path(db_path_stub)
    corpus_name = db_path_stub_obj.name  # e.g., "COHA" from "/path/to/COHA"
    corpus_path = db_path_stub_obj

    # Database name includes genre suffix if genre_focus is specified
    if genre_focus is not None:
        # Sort genres for consistent naming (e.g., "fic+mag" not "mag+fic")
        genre_suffix = "+".join(sorted(genre_focus))
        db_name = f"{corpus_name}_{genre_suffix}"
    else:
        db_name = corpus_name

    # Database path is in same directory as corpus
    db_path = corpus_path / db_name

    # Validate corpus path
    if not corpus_path.exists():
        raise ValueError(f"Corpus directory does not exist: {corpus_path}")

    text_dir = corpus_path / "text"
    if not text_dir.exists():
        raise ValueError(f"Text directory not found: {text_dir}")

    # Handle existing database - always remove for fresh start
    if db_path.exists():
        logger.info("Removing existing database for fresh start")
        # Use safe cleanup from ngramprep
        from ngramprep.ngram_acquire.utils.cleanup import safe_db_cleanup
        if not safe_db_cleanup(db_path):
            raise RuntimeError(
                f"Failed to remove existing database at {db_path}. "
                "Close open handles or remove it manually."
            )
        logger.info("Successfully removed existing database")

    # Determine worker count
    if workers is None:
        cpu_count = os.cpu_count() or 4
        workers = max(1, cpu_count - 1)

    # Discover text files
    logger.info("Discovering text files...")
    text_files = discover_text_files(text_dir)

    # Extract years from filenames
    file_year_pairs: List[Tuple[Path, int]] = []
    for text_file in text_files:
        try:
            year = extract_year_from_filename(text_file.name)
            file_year_pairs.append((text_file, year))
        except ValueError as e:
            logger.warning(f"Skipping file: {e}")
            continue

    # Print pipeline header
    print(format_banner(f"{corpus_name} CORPUS ACQUISITION", style="━"))
    print(f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}")
    print()
    print(format_banner("Configuration"))
    print(f"Corpus path:          {corpus_path}")
    print(f"Text directory:       {text_dir}")
    print(f"DB path:              {db_path}")
    print(f"Text files found:     {len(file_year_pairs)}")
    if genre_focus is not None:
        print(f"Genre focus:          {', '.join(genre_focus)}")
        print(f"Key format:           Year-only (training-ready)")
    else:
        print(f"Genre focus:          All genres")
        print(f"Key format:           Genre-prefixed (archival)")
    print(f"Workers:              {workers}")
    print(f"Batch size:           {write_batch_size:,}")
    print()
    print(format_banner("Processing Files"))
    sys.stdout.flush()

    # Open database and create writer
    logger.info("Opening database...")
    with open_db(db_path, profile="write:packed24", create_if_missing=True) as db:
        # Choose writer based on genre_focus parameter
        if genre_focus is None:
            # Archival mode: use genre-prefixed keys
            from .writer import SentenceBatchWriterWithGenre
            writer = SentenceBatchWriterWithGenre(db, batch_size=write_batch_size)
            use_genre_keys = True
        else:
            # Training mode: use year-only keys (genre already filtered)
            from .writer import SentenceBatchWriter
            writer = SentenceBatchWriter(db, batch_size=write_batch_size)
            use_genre_keys = False

        total_sentences = 0
        total_errors = 0
        files_processed = 0
        genre_stats = {}

        # Process files in parallel with progress bar
        with tqdm(
            total=len(file_year_pairs),
            desc="Files Processed",
            unit="files",
            ncols=100,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ) as pbar:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Submit all files for processing with worker IDs
                future_to_file = {
                    executor.submit(process_single_file, text_file, year, worker_id, combined_bigrams, genre_focus): (text_file, year)
                    for worker_id, (text_file, year) in enumerate(file_year_pairs)
                }

                # Process completed futures as they finish
                for future in as_completed(future_to_file):
                    text_file, year = future_to_file[future]
                    try:
                        filename, sentence_count, error_count, sentence_data, file_genre_stats = future.result()

                        # Write sentences to database (format depends on use_genre_keys)
                        for (genre, year, sentence_str), count in sentence_data.items():
                            tokens = sentence_str.split()
                            for _ in range(count):
                                if use_genre_keys:
                                    writer.add(year, tokens, genre)
                                else:
                                    writer.add(year, tokens)

                        # Update totals
                        total_sentences += sentence_count
                        files_processed += 1

                        # Merge genre stats
                        for genre, count in file_genre_stats.items():
                            genre_stats[genre] = genre_stats.get(genre, 0) + count

                    except Exception as e:
                        logger.error(f"Error processing {text_file.name}: {e}")
                        total_errors += 1
                    finally:
                        # Update progress bar
                        pbar.update(1)

        # Flush remaining sentences
        logger.info("Flushing remaining sentences...")
        writer.close()

        # Optional post-ingestion compaction
        if compact_after:
            _perform_compaction(db, db_path)

    # Print completion summary
    end_time = datetime.now()
    elapsed = end_time - start_time

    print("\nProcessing complete!")
    print()
    print(format_banner("Final Summary"))
    print(f"Files processed:          {files_processed}/{len(file_year_pairs)}")
    print(f"Failed files:             {total_errors}")
    print(f"Total sentences written:  {total_sentences:,}")
    print(f"Database path:            {db_path}")
    print()
    if genre_stats:
        print("Genre breakdown:")
        for genre, count in sorted(genre_stats.items()):
            print(f"  {genre:10s} {count:,} sentences")
        print()
    print(f"End Time: {end_time:%Y-%m-%d %H:%M:%S}")
    print(f"Total Runtime: {elapsed}")
    print()

    logger.info(f"Acquisition complete: {total_sentences:,} sentences written with genre metadata")
