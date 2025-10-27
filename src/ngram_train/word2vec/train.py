"""Main training orchestration for Word2Vec models."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product

from tqdm import tqdm

from .config import ensure_iterable, set_info
from .display import print_training_header, print_completion_banner, LINE_WIDTH
from .worker import train_model

__all__ = ["train_models"]


def train_models(
        corpus_path,
        years,  # Move required parameter before optional ones
        dir_suffix=None,
        weight_by=('freq',),
        vector_size=(100,),
        window=(2,),
        min_count=(1,),
        approach=('skip-gram',),
        epochs=(5,),
        max_parallel_models=os.cpu_count(),
        workers_per_model=1,
        allow_unk=False,
        max_unk_count=None,
        load_into_memory=True,
        shuffle=True,
        random_seed=42
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
        weight_by (tuple): Weighting strategies to try ("freq", "doc_freq", or "none").
        vector_size (tuple): Vector sizes to try.
        window (tuple): Window sizes to try.
        min_count (tuple): Minimum counts to try.
        approach (tuple): Training approaches to try ('CBOW' or 'skip-gram').
        epochs (tuple): Epoch counts to try.
        max_parallel_models (int): Maximum number of models to train in parallel.
        workers_per_model (int): Number of worker threads for each Word2Vec model.
        allow_unk (bool): Whether to allow ngrams with <UNK> tokens.
                         Recommended: False for Word2Vec (use only clean ngrams).
        max_unk_count (int): Maximum number of <UNK> tokens per ngram.
                            Only applies when allow_unk=True.
        load_into_memory (bool): If True, load all ngrams into memory before training.
                                Recommended: True (faster, enables shuffling).
        shuffle (bool): If True, shuffle ngrams before each epoch.
                       Recommended: True (avoids consecutive repetition).
        random_seed (int): Random seed for shuffling.
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

    # Format grid parameters
    year_range_str = f'{years[0]}–{years[1]} ({years[1] - years[0] + 1} years)'

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
        f"Allow <UNK>:          {allow_unk}",
        f"Max <UNK> count:      {max_unk_count}",
        f"Load into memory:     {load_into_memory}",
        f"Shuffle:              {shuffle}",
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

    tasks = [
        (year, db_path, model_dir, log_dir, params[0], params[1], params[2],
         params[3], params[4], params[5], workers_per_model, allow_unk, max_unk_count,
         load_into_memory, shuffle, random_seed)
        for year, params in product(years_range, param_combinations)
    ]

    # Calculate total tasks and print execution info
    param_combinations_count = len(param_combinations)
    years_count = years[1] - years[0] + 1
    total_tasks = years_count * param_combinations_count

    print("Execution")
    print("─" * LINE_WIDTH)
    print(f"Total models:         {total_tasks}")
    print(f"Parameter combos:     {param_combinations_count}")
    print(f"Years:                {years_count}")
    print("")

    # Simple aggregated progress bar showing completed models
    with tqdm(total=len(tasks), desc="Training Models", unit=" models") as pbar:
        with ProcessPoolExecutor(max_workers=max_parallel_models) as executor:
            futures = [executor.submit(train_model, *task) for task in tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"\nTask failed with error: {e}")
                pbar.update(1)

    # Print completion banner
    print_completion_banner(model_dir, total_tasks)
