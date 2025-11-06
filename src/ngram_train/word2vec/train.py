"""Main training orchestration for Word2Vec models."""

import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product

from gensim.models import KeyedVectors
from tqdm import tqdm

from .config import ensure_iterable, set_info
from .display import print_training_header, print_completion_banner, LINE_WIDTH
from .model import create_corpus_file
from .worker import train_model

__all__ = ["train_models"]


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

    for filename in tqdm(kv_files, desc="Scanning existing models", unit=" files"):
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
        weight_by=('freq',),
        vector_size=(100,),
        window=(2,),
        min_count=(1,),
        approach=('skip-gram',),
        epochs=(5,),
        max_parallel_models=os.cpu_count(),
        workers_per_model=1,
        unk_mode='reject',
        cache_corpus=False,
        use_corpus_file=True,
        temp_dir=None,
        debug_sample=0,
        debug_interval=0
):
    """
    Train Word2Vec models for multiple years from RocksDB.

    Args:
        corpus_path (str): Full path to corpus directory containing the database.
                          e.g., '/scratch/edk202/NLP_corpora/Google_Books/20200217/eng/5gram_files'
                          Must contain 'NLP_corpora' and end with 'Xgram_files'.
        years (tuple): Tuple of (start_year, end_year) inclusive.
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
        temp_dir (str): Optional directory for temporary corpus files. Useful for HPC scratch space
                       (e.g., '/scratch', '$TMPDIR', or os.environ.get('TMPDIR')). If None, uses
                       system default temp directory. Default: None.
        debug_sample (int): If > 0, print first N sentences for debugging (only for first model)
        debug_interval (int): If > 0, print one sample every N seconds (overrides debug_sample, only for first model)
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
        corpus_path, dir_suffix
    )

    # Handle directory management based on mode
    if mode == 'restart':
        # Remove existing directories completely
        if os.path.exists(model_dir):
            print(f"Restart mode: Removing existing models directory: {model_dir}")
            shutil.rmtree(model_dir)
        if os.path.exists(log_dir):
            print(f"Restart mode: Removing existing logs directory: {log_dir}")
            shutil.rmtree(log_dir)
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
        # Scan for existing valid models
        print("\nScanning for existing models...")
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

        # Ensure directories exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    else:
        raise ValueError(
            f"Invalid mode: '{mode}'. Must be 'resume', 'restart', or 'new'."
        )

    # Format grid parameters
    year_range_str = f'{years[0]}–{years[1]} ({years[1] - years[0] + 1} years)'

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

    print_training_header(
        start_time,
        db_path,
        model_dir,
        log_dir,
        max_parallel_models,
        grid_params
    )

    param_combinations = list(
        product(weight_by, vector_size, window, min_count, approach, epochs)
    )
    years_range = range(years[0], years[1] + 1)

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
    years_count = years[1] - years[0] + 1
    total_models_in_grid = len(all_tasks)
    models_to_train = len(tasks)

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

    # Create all corpus files in parallel if using corpus_file mode
    corpus_file_map = {}  # Maps (year, weight_by) -> corpus_file_path
    if use_corpus_file:
        print("Creating corpus files in parallel...")
        with ProcessPoolExecutor(max_workers=len(tasks_by_year_weight)) as executor:
            corpus_futures = {
                executor.submit(
                    create_corpus_file,
                    db_path=db_path,
                    year=year,
                    weight_by=weight_by_val,
                    unk_mode=unk_mode,
                    temp_dir=temp_dir
                ): (year, weight_by_val)
                for (year, weight_by_val) in tasks_by_year_weight.keys()
            }
            for future in as_completed(corpus_futures):
                year, weight_by_val = corpus_futures[future]
                try:
                    corpus_file_path = future.result()
                    corpus_file_map[(year, weight_by_val)] = corpus_file_path
                    print(f"Created corpus file for year {year}, weight_by={weight_by_val}: {corpus_file_path}")
                except Exception as e:
                    print(f"Failed to create corpus file for year {year}, weight_by={weight_by_val}: {e}")
                    raise
        print("")

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
    print(f"Tasks before grouping: {len(tasks)}")
    print(f"Tasks after regrouping: {len(all_tasks_to_run)}")
    print("")
    with tqdm(total=len(all_tasks_to_run), desc="Training Models", unit=" models") as pbar:
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
            # Clean up all corpus files
            if use_corpus_file:
                for (year, weight_by_val), corpus_file_path in corpus_file_map.items():
                    if corpus_file_path and os.path.exists(corpus_file_path):
                        try:
                            os.unlink(corpus_file_path)
                            tqdm.write(f"\nCleaned up corpus file: {corpus_file_path}")
                        except Exception as e:
                            tqdm.write(f"\nWarning: Could not remove corpus file {corpus_file_path}: {e}")

    # Print completion banner
    print_completion_banner(model_dir, models_trained)
