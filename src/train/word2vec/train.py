
"""Main training orchestration for Word2Vec models."""

from .display import truncate_path_to_fit, LINE_WIDTH

import os
import re
import shutil
import textwrap
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from typing import Optional, List

from gensim.models import KeyedVectors
from tqdm import tqdm

from .config import ensure_iterable, set_info
from .display import print_training_header, print_completion_banner
from .model import create_corpus_file
from .worker import train_model

__all__ = ["train_models", "build_word2vec_models", "transfer_models"]


def _parse_model_filename(filename):
    """
    Extract hyperparameters from model filename.

    Args:
        filename (str): Model filename (e.g., 'w2v_y2019_wbnone_vs300_w4_mc10_sg1_e15.kv')

    Returns:
        tuple or None: (year, weight_by, vector_size, window, min_count, sg, epochs)
                       or None if pattern doesn't match
    """
    pattern = r"w2v_y(\d+)_wb(\w+)_vs(\d{3})_w(\d{3})_mc(\d{3})_sg(\d+)_e(\d{3})\.kv"
    match = re.match(pattern, filename)
    if match:
        return (
            int(match.group(1)),   # year
            match.group(2),         # weight_by
            int(match.group(3)),   # vector_size
            int(match.group(4)),   # window
            int(match.group(5)),   # min_count
            int(match.group(6)),   # sg
            int(match.group(7))    # epochs
        )
    return None


def _is_model_valid(model_path):
    """
    Check if model file is complete and loadable.

    Args:
        model_path (str): Path to .kv model file

    Returns:
        bool: True if model is valid and complete, False otherwise
    """
    try:
        # Quick check: file exists and has reasonable size
        if not os.path.exists(model_path):
            return False
        if os.path.getsize(model_path) < 1000:  # Threshold for obviously corrupted files
            return False

        # Attempt to load - will fail for partial/corrupted files
        KeyedVectors.load(str(model_path))
        return True
    except Exception:
        # Any exception means the model is invalid
        return False


def _scan_existing_models(model_dir):
    """
    Scan model directory for valid existing models.

    Args:
        model_dir (str): Directory containing .kv model files

    Returns:
        tuple: (valid_set, invalid_list)
            - valid_set: set of parameter tuples for valid models
            - invalid_list: list of paths to invalid/partial models
    """
    valid_models = set()
    invalid_models = []

    if not os.path.exists(model_dir):
        return valid_models, invalid_models

    kv_files = [f for f in os.listdir(model_dir) if f.endswith('.kv')]

    if not kv_files:
        return valid_models, invalid_models

    for filename in tqdm(kv_files, desc="Scanning existing models", unit=" files", ncols=LINE_WIDTH):
        params = _parse_model_filename(filename)
        if params is None:
            continue  # Skip files that don't match expected pattern

        model_path = os.path.join(model_dir, filename)
        if _is_model_valid(model_path):
            valid_models.add(params)
        else:
            invalid_models.append(model_path)

    return valid_models, invalid_models


def _task_to_params(task):
    """
    Extract parameter tuple from task tuple for comparison with existing models.

    Args:
        task (tuple): Task tuple containing (year, db_path, model_dir, log_dir,
                      weight_by, vector_size, window, min_count, approach, epochs, ...)

    Returns:
        tuple: (year, weight_by, vector_size, window, min_count, sg, epochs)
    """
    year = task[0]
    weight_by = task[4]
    vector_size = task[5]
    window = task[6]
    min_count = task[7]
    approach = task[8]
    epochs = task[9]

    # Convert approach to sg value (same logic as in worker.py)
    sg = 1 if approach == 'skip-gram' else 0

    return (year, weight_by, vector_size, window, min_count, sg, epochs)


def train_models(
        corpus_path,
        years,  # Move required parameter before optional ones
        dir_suffix=None,
        mode='resume',
        year_step=1,
        weight_by=('freq',),
        vector_size=(100,),
        window=(2,),
        min_count=(1,),
        approach=('skip-gram',),
        epochs=(5,),
        max_parallel_models=os.cpu_count(),
        max_corpus_workers=8,
        workers_per_model=1,
        unk_mode='reject',
        cache_corpus=False,
        use_corpus_file=True,
        reuse_corpus_files=False,
        temp_dir=None,
        debug_sample=0,
        debug_interval=0,
        genre_focus: Optional[List[str]] = None
):
    """
    Train Word2Vec models for multiple years from RocksDB.

    Args:
        corpus_path (str): Full path to corpus directory containing the database.
                          For ngrams: e.g., '/scratch/edk202/NLP_corpora/Google_Books/.../5gram_files'
                          For Davies: e.g., '/scratch/edk202/NLP_corpora/COHA'
        years (tuple): Tuple of (start_year, end_year) inclusive.
        year_step (int): Step size for year increments. Default: 1 (every year).
                        Examples: year_step=5 trains (1900, 1905, 1910, ...),
                                 year_step=10 trains (1900, 1910, 1920, ...)
        dir_suffix (str): Suffix for model and log directories (e.g., 'window_comparison').
                         If None, generates timestamp-based name (e.g., '20241027_143022').
                         Recommended: Use descriptive names for experiments.
        mode (str): Training mode - one of:
                   'resume' (default): Skip existing valid models, retrain partial/corrupted ones
                   'restart': Erase model and log directories, start from scratch
                   'new': Fail if directories exist (safety check for new experiments)
        weight_by (tuple): Weighting strategies to try ("freq", "doc_freq", or "none").
        vector_size (tuple): Vector sizes to try.
        window (tuple): Window sizes to try.
        min_count (tuple): Minimum counts to try.
        approach (tuple): Training approaches to try ('CBOW' or 'skip-gram').
        epochs (tuple): Epoch counts to try.
        max_parallel_models (int): Maximum number of models to train in parallel.
        max_corpus_workers (int): Maximum number of parallel workers for corpus file creation.
                                 Default: 8. Reduce if OOM errors occur during corpus creation.
        workers_per_model (int): Number of worker threads for each Word2Vec model.
                                When use_corpus_file=True, can scale to 32+ workers on HPC.
                                When use_corpus_file=False, optimal range is 8-12 workers.
        unk_mode (str): How to handle <UNK> tokens. One of:
                       - 'reject': Discard entire n-gram if it contains any <UNK> (default)
                       - 'strip': Remove <UNK> tokens, keep if ≥2 tokens remain
                       - 'retain': Keep n-grams as-is, including <UNK> tokens
        cache_corpus (bool): If True, load entire corpus into memory before training.
                            Significantly speeds up multi-epoch training by eliminating disk I/O
                            after first epoch. Requires sufficient RAM. Only applies when
                            use_corpus_file=False. Default: False.
        use_corpus_file (bool): If True, stream corpus to temporary file and use corpus_file
                               parameter for training. Enables better multi-core scaling (32+ workers)
                               by bypassing Python's GIL. If False, use iterator-based approach
                               (optimal for 8-12 workers). Default: True.
        reuse_corpus_files (bool): If True and use_corpus_file=True, preserve corpus files in
                                  temp_dir across runs and reuse them instead of recreating.
                                  Useful for resuming training without regenerating corpus files.
                                  Default: False.
        temp_dir (str): Optional directory for temporary corpus files. Useful for HPC scratch space
                       (e.g., '/scratch', '$TMPDIR', or os.environ.get('TMPDIR')). If None, uses
                       system default temp directory. Default: None.
        debug_sample (int): If > 0, print first N sentences for debugging (only for first model)
        debug_interval (int): If > 0, print one sample every N seconds (overrides debug_sample, only for first model)
        genre_focus (list): Optional list of genres for Davies corpora (e.g., ['fic']).
                           Used to locate the correct filtered database (e.g., COHA_fic_filtered).
                           If None, looks for plain corpus database (e.g., COHA_filtered).
    """
    # Generate default suffix if not provided
    if dir_suffix is None:
        dir_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"No dir_suffix provided. Using timestamp: {dir_suffix}\n")

    weight_by = ensure_iterable(weight_by)
    vector_size = ensure_iterable(vector_size)
    window = ensure_iterable(window)
    min_count = ensure_iterable(min_count)
    approach = ensure_iterable(approach)
    epochs = ensure_iterable(epochs)

    start_time, db_path, model_dir, log_dir = set_info(
        corpus_path, dir_suffix, genre_focus
    )

    # Handle directory management based on mode
    if mode == 'restart':
        # Remove existing directories completely
        def robust_rmtree(path, max_retries=5):
            """Robustly remove directory tree, handling NFS/filesystem issues."""
            import time

            for attempt in range(max_retries):
                try:
                    shutil.rmtree(path, ignore_errors=False)
                    return
                except OSError as e:
                    error_msg = str(e)
                    # Check if it's an NFS lock issue
                    if '.nfs' in error_msg or 'Device or resource busy' in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = 1.0 * (attempt + 1)  # Increasing backoff
                            time.sleep(wait_time)
                        else:
                            # On final attempt, use ignore_errors to remove what we can
                            shutil.rmtree(path, ignore_errors=True)
                            return
                    else:
                        # For non-NFS errors, retry with shorter delay
                        if attempt < max_retries - 1:
                            time.sleep(0.5)
                        else:
                            raise

        if os.path.exists(model_dir):
            robust_rmtree(model_dir)
        if os.path.exists(log_dir):
            robust_rmtree(log_dir)

        # Remove evaluation results file if it exists
        model_base = os.path.dirname(model_dir)
        eval_file = os.path.join(model_base, f"evaluation_results_{dir_suffix}.csv")
        if os.path.exists(eval_file):
            os.remove(eval_file)

        # Recreate directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        existing_valid = set()
        invalid_models = []
        print("")

    elif mode == 'new':
        # Fail if directories exist (safety check)
        if os.path.exists(model_dir) or os.path.exists(log_dir):
            raise FileExistsError(
                f"Mode 'new' requires non-existent directories.\n"
                f"Found existing: {model_dir if os.path.exists(model_dir) else log_dir}\n"
                f"Use mode='resume' to continue or mode='restart' to erase and start over."
            )
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        existing_valid = set()
        invalid_models = []

    elif mode == 'resume':
        # Ensure directories exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        # Will scan for existing models after printing the banner

    else:
        raise ValueError(
            f"Invalid mode: '{mode}'. Must be 'resume', 'restart', or 'new'."
        )

    # Handle temp_dir setup
    if temp_dir:
        temp_dir = os.path.abspath(os.path.expanduser(temp_dir))
        if not reuse_corpus_files and os.path.exists(temp_dir):
            # Clear contents unless reuse is enabled
            for entry in os.listdir(temp_dir):
                entry_path = os.path.join(temp_dir, entry)
                try:
                    if os.path.isdir(entry_path):
                        shutil.rmtree(entry_path, ignore_errors=False)
                    else:
                        os.unlink(entry_path)
                except OSError:
                    pass  # Skip NFS lock files (.nfs*) and other busy resources
        os.makedirs(temp_dir, exist_ok=True)

    # Format grid parameters with step information
    total_years = len(range(years[0], years[1] + 1, year_step))
    if year_step == 1:
        year_range_str = f'{years[0]}–{years[1]} ({total_years} years)'
    else:
        year_range_str = f'{years[0]}–{years[1]} (step={year_step}, {total_years} years)'

    # Determine corpus mode description
    if use_corpus_file:
        corpus_mode = "Shared corpus file (one per year/weight)"
    elif cache_corpus:
        corpus_mode = "Memory cache (per model)"
    else:
        corpus_mode = "Streaming iterator (per model)"

    grid_params = '\n'.join([
        "Training Parameters",
        "─" * LINE_WIDTH,
        f"Years:                {year_range_str}",
        f"Weighting:            {weight_by}",
        f"Vector size:          {vector_size}",
        f"Context window:       {window}",
        f"Minimum word count:   {min_count}",
        f"Approach:             {approach}",
        f"Training epochs:      {epochs}",
        "",
        "Data Options",
        "─" * LINE_WIDTH,
        f"UNK mode:             {unk_mode}",
        f"Corpus mode:          {corpus_mode}",
        f"Workers per model:    {workers_per_model}",
    ])

    # Attach temp_dir to print_training_header for display
    if temp_dir:
        print_training_header.temp_dir = temp_dir
    else:
        print_training_header.temp_dir = None
    print_training_header(
        start_time,
        db_path,
        model_dir,
        log_dir,
        max_parallel_models,
        grid_params
    )

    # Scan for existing models in resume mode (after banner)
    if mode == 'resume':
        existing_valid, invalid_models = _scan_existing_models(model_dir)
        print(f"  Valid models found:    {len(existing_valid)}")
        print(f"  Invalid/partial:       {len(invalid_models)}")

        # Remove invalid models to retrain them
        if invalid_models:
            print(f"\nRemoving {len(invalid_models)} invalid/partial model files...")
            for path in invalid_models:
                try:
                    os.remove(path)
                    print(f"  Removed: {os.path.basename(path)}")
                except Exception as e:
                    print(f"  Warning: Could not remove {os.path.basename(path)}: {e}")
            print("")
    else:
        existing_valid = set()
        invalid_models = []

    param_combinations = list(
        product(weight_by, vector_size, window, min_count, approach, epochs)
    )
    years_range = range(years[0], years[1] + 1, year_step)

    # Build full task list (for statistics and filtering)
    all_tasks = [
        (year, db_path, model_dir, log_dir, params[0], params[1], params[2],
         params[3], params[4], params[5], workers_per_model, unk_mode, cache_corpus,
         use_corpus_file, None, temp_dir,  # corpus_file_path=None, will be set later
         debug_sample if idx == 0 else 0, debug_interval if idx == 0 else 0)
        for idx, (year, params) in enumerate(product(years_range, param_combinations))
    ]

    # Filter tasks based on existing models in resume mode
    if mode == 'resume' and existing_valid:
        tasks = [
            task for task in all_tasks
            if _task_to_params(task) not in existing_valid
        ]
        skipped_count = len(all_tasks) - len(tasks)
    else:
        tasks = all_tasks
        skipped_count = 0

    # Calculate statistics
    param_combinations_count = len(param_combinations)
    years_count = len(years_range)
    total_models_in_grid = len(all_tasks)
    models_to_train = len(tasks)

    print("")
    print("Execution")
    print("─" * LINE_WIDTH)
    print(f"Total models in grid: {total_models_in_grid}")
    if mode == 'resume' and skipped_count > 0:
        print(f"Existing valid:       {len(existing_valid)}")
        print(f"Models to train:      {models_to_train}")
    else:
        print(f"Models to train:      {models_to_train}")
    print(f"Parameter combos:     {param_combinations_count}")
    print(f"Years:                {years_count}")
    print("")

    # Group tasks by (year, weight_by) for shared corpus file creation
    # Key insight: Different weight_by strategies need different corpus files
    tasks_by_year_weight = defaultdict(list)
    for task in tasks:
        year = task[0]
        weight_by_val = task[4]
        tasks_by_year_weight[(year, weight_by_val)].append(task)

    # Create corpus files if using corpus_file mode, reusing cached files
    corpus_file_map = {}  # Maps (year, weight_by) -> corpus_file_path
    if use_corpus_file and tasks_by_year_weight:
        # Scan for cached corpus files in temp_dir
        needed = set(tasks_by_year_weight.keys())
        if reuse_corpus_files and temp_dir and os.path.exists(temp_dir):
            corpus_pattern = re.compile(r'^w2v_corpus_y(\d+)_wb(\w+)_.+\.txt$')
            for entry in os.listdir(temp_dir):
                m = corpus_pattern.match(entry)
                if m:
                    key = (int(m.group(1)), m.group(2))
                    if key in needed and key not in corpus_file_map:
                        corpus_file_map[key] = os.path.join(temp_dir, entry)

        still_needed = needed - corpus_file_map.keys()
        if corpus_file_map:
            print(f"Reusing {len(corpus_file_map)} cached corpus files", flush=True)
        if still_needed:
            print(f"Creating {len(still_needed)} corpus files in parallel...", flush=True)
            n_corpus_workers = min(len(still_needed), max_corpus_workers)
            with ProcessPoolExecutor(max_workers=n_corpus_workers) as executor:
                corpus_futures = {
                    executor.submit(
                        create_corpus_file,
                        db_path=db_path,
                        year=year,
                        weight_by=weight_by_val,
                        unk_mode=unk_mode,
                        temp_dir=temp_dir
                    ): (year, weight_by_val)
                    for (year, weight_by_val) in still_needed
                }
                created_years = []
                failed_years = []
                for future in as_completed(corpus_futures):
                    year, weight_by_val = corpus_futures[future]
                    try:
                        corpus_file_path = future.result()
                        corpus_file_map[(year, weight_by_val)] = corpus_file_path
                        created_years.append(year)
                    except Exception as e:
                        failed_years.append(year)
                        print(f"Failed to create corpus file for year {year}, weight_by={weight_by_val}: {e}")
                        raise
                if created_years:
                    msg = f"Corpus files created for years: {sorted(set(created_years))}"
                    print(textwrap.fill(msg, width=LINE_WIDTH, subsequent_indent='  '))
                if failed_years:
                    msg = f"Corpus file creation failed for years: {sorted(set(failed_years))}"
                    print(textwrap.fill(msg, width=LINE_WIDTH, subsequent_indent='  '))
        print("")  # Blank line after corpus file creation

    # Update all tasks to include their corpus_file_path
    if use_corpus_file:
        updated_tasks_by_year_weight = {}
        for (year, weight_by_val), year_tasks in tasks_by_year_weight.items():
            corpus_file_path = corpus_file_map.get((year, weight_by_val))
            if corpus_file_path:
                updated_tasks = [
                    task[:14] + (corpus_file_path,) + task[15:]
                    for task in year_tasks
                ]
                updated_tasks_by_year_weight[(year, weight_by_val)] = updated_tasks
            else:
                updated_tasks_by_year_weight[(year, weight_by_val)] = year_tasks
        tasks_by_year_weight = updated_tasks_by_year_weight

    # Train all models
    models_trained = 0
    all_tasks_to_run = [task for year_tasks in tasks_by_year_weight.values() for task in year_tasks]
    with tqdm(total=len(all_tasks_to_run), desc="Training Models", unit=" models", ncols=LINE_WIDTH) as pbar:
        try:
            # Train all models across all years in parallel
            with ProcessPoolExecutor(max_workers=max_parallel_models) as executor:
                futures = [executor.submit(train_model, *task) for task in all_tasks_to_run]
                for future in as_completed(futures):
                    try:
                        future.result()
                        models_trained += 1
                    except Exception as e:
                        tqdm.write(f"\nTask failed with error: {e}")
                    pbar.update(1)
        finally:
            # Clean up corpus files unless reuse is enabled
            if use_corpus_file and not reuse_corpus_files:
                for (year, weight_by_val), corpus_file_path in corpus_file_map.items():
                    if corpus_file_path and os.path.exists(corpus_file_path):
                        try:
                            os.unlink(corpus_file_path)
                        except Exception:
                            pass  # Silently ignore cleanup errors

                # Clean up temp directory after all processing is complete
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass  # Silently ignore cleanup errors

    # Print completion banner
    print_completion_banner(model_dir, models_trained)


def _setup_logging(log_dir):
    """
    Set up logging for the training pipeline if not already configured.

    Args:
        log_dir: Directory to store log files
    """
    import logging

    # Check if logging is already configured
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        # Logging already configured, don't override
        return

    # Set up logging to log directory
    from ngramprep.ngram_acquire.logger import setup_logger

    log_file = setup_logger(
        db_path=str(log_dir),
        filename_prefix="word2vec_training",
        console=False,
        rotate=True,
        max_bytes=100_000_000,
        backup_count=5,
        force=False  # Don't override if already configured
    )

    # Log initial context
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Word2Vec Training Pipeline")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)


def build_word2vec_models(
    ngram_size=None,
    repo_release_id=None,
    repo_corpus_id=None,
    db_path_stub=None,
    years=None,
    dir_suffix=None,
    mode='resume',
    year_step=1,
    weight_by=('freq',),
    vector_size=(100,),
    window=(2,),
    min_count=(1,),
    approach=('skip-gram',),
    epochs=(5,),
    max_parallel_models=None,
    max_corpus_workers=8,
    workers_per_model=1,
    unk_mode='reject',
    cache_corpus=False,
    use_corpus_file=True,
    reuse_corpus_files=False,
    temp_dir=None,
    debug_sample=0,
    debug_interval=0
):
    """
    Convenience wrapper for training Word2Vec models using path stub parameters.

    This function mirrors the interface of build_processed_db and build_pivoted_db,
    taking simple path parameters instead of requiring manual path construction.

    Args:
        ngram_size (int): N-gram size (e.g., 5 for 5grams) - required
        repo_release_id (str): Release date in YYYYMMDD format (e.g., "20200217") - required
        repo_corpus_id (str): Corpus identifier (e.g., "eng", "eng-fiction") - required
        db_path_stub (str): Base directory for data (e.g., "/scratch/edk202/NLP_corpora/Google_Books/") - required
        years (tuple): Tuple of (start_year, end_year) inclusive - required
        dir_suffix (str): Suffix for model/log directories (e.g., 'test', 'final'). If None, uses timestamp.
        mode (str): Training mode - 'resume', 'restart', or 'new'
        year_step (int): Step size for year increments (default: 1)
        weight_by (tuple): Weighting strategies ("freq", "doc_freq", or "none")
        vector_size (tuple): Vector sizes to try
        window (tuple): Window sizes to try
        min_count (tuple): Minimum counts to try
        approach (tuple): Training approaches ('CBOW' or 'skip-gram')
        epochs (tuple): Epoch counts to try
        max_parallel_models (int): Maximum parallel models (default: cpu_count())
        max_corpus_workers (int): Maximum parallel workers for corpus file creation (default: 8)
        workers_per_model (int): Worker threads per model
        unk_mode (str): How to handle <UNK> tokens ('reject', 'strip', or 'retain')
        cache_corpus (bool): Load entire corpus into memory
        use_corpus_file (bool): Use temporary corpus files for better scaling
        reuse_corpus_files (bool): Preserve and reuse corpus files in temp_dir across runs
        temp_dir (str): Directory for temporary files
        debug_sample (int): Print first N sentences for debugging
        debug_interval (int): Print samples every N seconds

    Returns:
        str: Path to the models directory

    Example:
        >>> build_word2vec_models(
        ...     ngram_size=5,
        ...     repo_release_id='20200217',
        ...     repo_corpus_id='eng-fiction',
        ...     db_path_stub='/scratch/edk202/NLP_corpora/Google_Books/',
        ...     years=(1900, 2019),
        ...     dir_suffix='final',
        ...     vector_size=(200,),
        ...     window=(4,),
        ...     epochs=(10,),
        ...     approach=('skip-gram',)
        ... )
    """
    # Validate required parameters
    if ngram_size is None or repo_release_id is None or repo_corpus_id is None or db_path_stub is None:
        raise ValueError(
            "All path stub parameters are required: "
            "ngram_size, repo_release_id, repo_corpus_id, db_path_stub"
        )

    if years is None:
        raise ValueError("years parameter is required (tuple of start_year, end_year)")

    # Construct corpus path from stub parameters
    from ngramprep.ngram_acquire.db.build_path import build_db_path
    from pathlib import Path

    base_path = Path(build_db_path(db_path_stub, ngram_size, repo_release_id, repo_corpus_id)).parent
    corpus_path = str(base_path)

    # Set default max_parallel_models if not provided
    if max_parallel_models is None:
        max_parallel_models = os.cpu_count()

    # Generate default suffix if not provided
    if dir_suffix is None:
        dir_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"No dir_suffix provided. Using timestamp: {dir_suffix}\n")

    # Set up logging internally
    # We need to construct log_dir first to set up logging
    from .config import construct_model_path
    model_base = construct_model_path(corpus_path)
    log_dir = os.path.join(model_base, f"logs_{dir_suffix}", "training")
    os.makedirs(log_dir, exist_ok=True)
    _setup_logging(log_dir)

    # Call the main training function
    train_models(
        corpus_path=corpus_path,
        years=years,
        dir_suffix=dir_suffix,
        mode=mode,
        year_step=year_step,
        weight_by=weight_by,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        approach=approach,
        epochs=epochs,
        max_parallel_models=max_parallel_models,
        max_corpus_workers=max_corpus_workers,
        workers_per_model=workers_per_model,
        unk_mode=unk_mode,
        cache_corpus=cache_corpus,
        use_corpus_file=use_corpus_file,
        reuse_corpus_files=reuse_corpus_files,
        temp_dir=temp_dir,
        debug_sample=debug_sample,
        debug_interval=debug_interval
    )

    # Return model directory path
    model_dir = os.path.join(model_base, f"models_{dir_suffix}")
    return model_dir


def transfer_models(
    source_suffix,
    dest_suffix,
    # Google Ngrams pipeline parameters
    ngram_size=None,
    repo_release_id=None,
    repo_corpus_id=None,
    db_path_stub=None,
    # Davies pipeline parameters
    corpus_path=None,
    genre_focus=None,
    # Common parameters
    filter_params=None,
    overwrite=False,
    validate=True,
    verbose=True
):
    """
    Transfer models with specific hyperparameters from one dir_suffix to another.

    This function is useful when you've completed hyperparameter grid search and want
    to copy selected models (e.g., those with optimal hyperparameters) to a different
    directory for final use.

    Supports two calling conventions (mutually exclusive):

    **Google Ngrams pipeline** — supply all four ngram params:
        ngram_size, repo_release_id, repo_corpus_id, db_path_stub

    **Davies pipeline** — supply corpus_path (and optionally genre_focus):
        corpus_path, genre_focus

    Args:
        source_suffix (str): Source dir_suffix (e.g., 'test').
        dest_suffix (str): Destination dir_suffix (e.g., 'final').
        ngram_size (int, optional): Size of n-grams (e.g., 5). Ngrams pipeline only.
        repo_release_id (str, optional): Release identifier (e.g., '20200217'). Ngrams pipeline only.
        repo_corpus_id (str, optional): Corpus identifier (e.g., 'eng'). Ngrams pipeline only.
        db_path_stub (str, optional): Base path to corpus database directory. Ngrams pipeline only.
        corpus_path (str, optional): Full path to Davies corpus directory (e.g.,
            '/scratch/edk202/NLP_corpora/COHA'). Davies pipeline only.
        genre_focus (list, optional): List of genres for Davies corpora (e.g., ['fic']).
            Davies pipeline only. If None, no genre subdirectory is used.
        filter_params (dict, optional): Dictionary of hyperparameter filters.
            Keys can be: 'year', 'weight_by', 'vector_size', 'window',
            'min_count', 'sg', 'epochs'. Values can be single values or tuples/lists
            for multiple matches. Examples:
            - {'vector_size': 300, 'epochs': 10}
            - {'vector_size': (200, 300), 'epochs': 10}
            If None, all models are transferred.
        overwrite (bool): If True, remove existing destination directory before transfer.
            If False and destination exists, raises error. Default: False.
        validate (bool): If True, only transfer models that pass validation check.
            Default: True.
        verbose (bool): If True, print detailed progress information. Default: True.

    Returns:
        dict: Summary with keys:
            - 'transferred': Number of models successfully transferred
            - 'skipped': Number of models skipped (failed validation)
            - 'total_found': Total models matching filter
            - 'source_dir': Source directory path
            - 'dest_dir': Destination directory path

    Raises:
        ValueError: If source directory doesn't exist or contains no models, or if
            neither pipeline's required parameters are supplied.
        FileExistsError: If destination exists and overwrite=False.

    Example (Ngrams pipeline):
        >>> result = transfer_models(
        ...     source_suffix='test',
        ...     dest_suffix='final',
        ...     ngram_size=5,
        ...     repo_release_id='20200217',
        ...     repo_corpus_id='eng',
        ...     db_path_stub='/scratch/edk202/NLP_corpora/Google_Books/',
        ...     filter_params={'vector_size': 300, 'epochs': 10},
        ...     overwrite=True
        ... )

    Example (Davies pipeline):
        >>> result = transfer_models(
        ...     source_suffix='test',
        ...     dest_suffix='final',
        ...     corpus_path='/scratch/edk202/NLP_corpora/COHA',
        ...     filter_params={'vector_size': 100, 'epochs': 10},
        ...     overwrite=True
        ... )
    """
    from .config import construct_model_path
    from .display import truncate_path_to_fit

    # Construct source and destination paths
    ngram_params = (ngram_size, repo_release_id, repo_corpus_id, db_path_stub)
    if corpus_path is not None:
        # Davies pipeline: build model path from corpus_path + optional genre subdir
        corpus_path = corpus_path.rstrip('/')
        corpus_name = os.path.basename(corpus_path)
        if genre_focus is not None:
            genre_suffix = "+".join(sorted(genre_focus))
            genre_subdir = f"{corpus_name}_{genre_suffix}"
        else:
            genre_subdir = corpus_name
        model_base = os.path.join(construct_model_path(corpus_path), genre_subdir)
    elif all(p is not None for p in ngram_params):
        # Ngrams pipeline: build model path from release/corpus/ngram hierarchy
        _corpus_path = os.path.join(
            db_path_stub, repo_release_id, repo_corpus_id, f"{ngram_size}gram_files"
        )
        model_base = construct_model_path(_corpus_path)
    else:
        raise ValueError(
            "Supply either corpus_path (Davies pipeline) or all of "
            "ngram_size, repo_release_id, repo_corpus_id, and db_path_stub "
            "(Ngrams pipeline)."
        )

    source_dir = os.path.join(model_base, f"models_{source_suffix}")
    dest_dir = os.path.join(model_base, f"models_{dest_suffix}")

    # Validate source directory exists
    if not os.path.exists(source_dir):
        raise ValueError(f"Source directory does not exist: {source_dir}")

    # Check for models in source
    source_files = [f for f in os.listdir(source_dir) if f.endswith('.kv')]
    if not source_files:
        raise ValueError(f"No .kv model files found in source directory: {source_dir}")

    # Handle destination directory
    if os.path.exists(dest_dir):
        if not overwrite:
            raise FileExistsError(
                f"Destination directory already exists: {dest_dir}\n"
                f"Use overwrite=True to remove existing directory."
            )
        shutil.rmtree(dest_dir)

    os.makedirs(dest_dir, exist_ok=True)

    if verbose:
        source_str = truncate_path_to_fit(source_dir, "Source:              ", LINE_WIDTH)
        dest_str = truncate_path_to_fit(dest_dir, "Destination:         ", LINE_WIDTH)
        filter_str = str(filter_params) if filter_params else "None (transferring all models)"

        print("\nMODEL TRANSFER")
        print("━" * LINE_WIDTH)
        print(f"Source:              {source_str}")
        print(f"Destination:         {dest_str}")
        print(f"Filters:             {filter_str}")
        print(f"Validate:            {validate}")
        print(f"Overwrite:           {overwrite}")
        print("")

    # Normalize filter_params values to sets for easy matching
    normalized_filters = {}
    if filter_params:
        for key, value in filter_params.items():
            if isinstance(value, (list, tuple)):
                normalized_filters[key] = set(value)
            else:
                normalized_filters[key] = {value}

    # Filter and transfer models
    transferred = 0
    skipped = 0
    total_found = 0

    iterator = (
        tqdm(source_files, desc="Transferring models", ncols=LINE_WIDTH, unit=" files")
        if verbose else source_files
    )

    for filename in iterator:
        params = _parse_model_filename(filename)
        if params is None:
            continue

        year, weight_by, vector_size, window, min_count, sg, epochs = params

        if normalized_filters:
            param_dict = {
                'year': year,
                'weight_by': weight_by,
                'vector_size': vector_size,
                'window': window,
                'min_count': min_count,
                'sg': sg,
                'epochs': epochs
            }
            if not all(param_dict.get(k) in v for k, v in normalized_filters.items()):
                continue

        total_found += 1
        source_path = os.path.join(source_dir, filename)

        if validate and not _is_model_valid(source_path):
            skipped += 1
            if verbose:
                tqdm.write(f"  Skipping invalid model: {filename}")
            continue

        dest_path = os.path.join(dest_dir, filename)
        try:
            shutil.copy2(source_path, dest_path)
            transferred += 1
        except Exception as e:
            skipped += 1
            if verbose:
                tqdm.write(f"  Error copying {filename}: {e}")

    if verbose:
        dest_str = truncate_path_to_fit(dest_dir, "Destination:         ", LINE_WIDTH)
        print("\nTransfer Complete")
        print("═" * LINE_WIDTH)
        print(f"Models found:        {total_found}")
        print(f"Transferred:         {transferred}")
        print(f"Skipped:             {skipped}")
        print(f"Destination:         {dest_str}")
        print("━" * LINE_WIDTH)
        print("")

    return {
        'transferred': transferred,
        'skipped': skipped,
        'total_found': total_found,
        'source_dir': source_dir,
        'dest_dir': dest_dir
    }
