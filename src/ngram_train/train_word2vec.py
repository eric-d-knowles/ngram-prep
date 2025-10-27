"""
Word2Vec Training Module for Google N-grams

Trains Word2Vec models on Google n-gram data stored in RocksDB.
Supports streaming or memory-loaded modes with optional shuffling.
"""

import logging
import os
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product, repeat
from math import log, floor

from gensim.models import Word2Vec
from tqdm import tqdm

try:
    import enlighten
except ImportError:
    enlighten = None

try:
    from setproctitle import setproctitle
except ImportError:
    setproctitle = None

from ngram_prep.ngram_pivot.stream import stream_year_ngrams


def ensure_iterable(param):
    """
    Ensure the input parameter is iterable (e.g., a tuple).
    """
    return param if isinstance(param, (tuple, list)) else (param,)


def construct_model_path(corpus_path):
    """
    Construct the parallel model path from a corpus path.

    Replaces 'NLP_corpora' with 'NLP_models' in the path.

    Args:
        corpus_path (str): Path to corpus (e.g., '/scratch/edk202/NLP_corpora/Google_Books/...')

    Returns:
        str: Parallel model path (e.g., '/scratch/edk202/NLP_models/Google_Books/...')

    Raises:
        ValueError: If 'NLP_corpora' not found in path.
    """
    if 'NLP_corpora' not in corpus_path:
        raise ValueError(
            f"corpus_path must contain 'NLP_corpora' but got: {corpus_path}"
        )

    model_path = corpus_path.replace('NLP_corpora', 'NLP_models')
    return model_path


def set_info(corpus_path, dir_suffix):
    """
    Set up project paths for database, models, and logs.

    Args:
        corpus_path (str): Full path to corpus directory containing the database.
                          e.g., '/scratch/edk202/NLP_corpora/Google_Books/20200217/eng/5gram_files'
        dir_suffix (str): Suffix for model and log directories.

    Returns:
        tuple: Start time, database path, model directory, log directory.
    """
    start_time = datetime.now()

    # Extract ngram size from corpus path (e.g., '5gram_files' -> 5)
    basename = os.path.basename(corpus_path)
    if 'gram_files' not in basename:
        raise ValueError(
            f"corpus_path must end with 'Xgram_files' but got: {corpus_path}"
        )
    ngram_size = basename.replace('gram_files', '')

    # Database path
    db_path = os.path.join(corpus_path, f"{ngram_size}grams_pivoted.db")

    # Construct parallel model path
    model_base = construct_model_path(corpus_path)

    # Model directory
    model_dir = os.path.join(model_base, f"models_{dir_suffix}")

    # Log directory
    log_dir = os.path.join(model_base, f"logs_{dir_suffix}", "training")

    return start_time, db_path, model_dir, log_dir, ngram_size


def print_info(
        start_time,
        db_path,
        model_dir,
        log_dir,
        ngram_size,
        max_parallel_models,
        grid_params
):
    """
    Print project setup information.

    Args:
        start_time (datetime): Start time of the process.
        db_path (str): Database path.
        model_dir (str): Model directory path.
        log_dir (str): Log directory path.
        ngram_size (str): The size of ngrams.
        max_parallel_models (int): Number of parallel models.
        grid_params (str): Grid search parameters.
    """
    print(f"\033[31mStart Time:         {start_time}\n\033[0m")
    print("\033[4mTraining Info\033[0m")
    print(f"Ngram size:         {ngram_size}")
    print(f"Database path:      {db_path}")
    print(f"Model directory:    {model_dir}")
    print(f"Log directory:      {log_dir}")
    print(f"Parallel models:    {max_parallel_models}\n")
    print("Grid parameters:")
    print(f"{grid_params}\n")


def calculate_weight(freq, base=10):
    """
    Calculate the weight of an n-gram using logarithmic scaling.

    Args:
        freq (int): Raw frequency of the n-gram.
        base (float): Logarithm base for scaling.

    Returns:
        int: Scaled weight.
    """
    return max(1, floor(log(freq + 1, base)))


def create_unk_filter(allow_unk=True, max_unk_count=None):
    """
    Create a filter function for handling <UNK> tokens in ngrams.

    Args:
        allow_unk (bool): Whether to allow ngrams with <UNK> tokens at all.
        max_unk_count (int): Maximum number of <UNK> tokens per ngram (None = no limit).

    Returns:
        callable or None: Filter function for stream_year_ngrams, or None if no filtering.
    """
    if allow_unk and max_unk_count is None:
        return None  # No filtering needed

    def filter_fn(ngram_bytes):
        if not allow_unk and b'<UNK>' in ngram_bytes:
            return False  # Skip ngrams with any <UNK>
        if max_unk_count is not None:
            unk_count = ngram_bytes.count(b'<UNK>')
            return unk_count <= max_unk_count
        return True

    return filter_fn


class SentencesIterable:
    """
    An iterable wrapper for sentences generated from RocksDB pivoted ngram database.
    Supports streaming, loading into memory, and optional shuffling.
    """

    def __init__(self, db_path, year, weight_by="freq", log_base=10,
                 allow_unk=True, max_unk_count=None,
                 load_into_memory=False, shuffle=False, random_seed=42,
                 progress_position=None):
        """
        Initialize the iterable.

        Args:
            db_path (str): Path to RocksDB database.
            year (int): Year to load data for.
            weight_by (str): Weighting strategy ("freq", "doc_freq", or "none").
            log_base (int): Base for logarithmic weighting.
            allow_unk (bool): Whether to allow ngrams with <UNK> tokens.
            max_unk_count (int): Maximum number of <UNK> tokens per ngram.
            load_into_memory (bool): If True, load all ngrams into memory.
                                    If False, stream from database each iteration.
            shuffle (bool): If True, shuffle ngrams before each epoch.
                           Only applies when load_into_memory=True.
            random_seed (int): Random seed for shuffling (only used if shuffle=True).
            progress_position (int): Position for tqdm progress bar (for parallel loading).
        """
        self.db_path = db_path
        self.year = year
        self.weight_by = weight_by
        self.log_base = log_base
        self.ngram_filter = create_unk_filter(allow_unk, max_unk_count)
        self.load_into_memory = load_into_memory
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.progress_position = progress_position

        # Validate parameters
        if shuffle and not load_into_memory:
            raise ValueError("shuffle=True requires load_into_memory=True")

        # Cache for loaded data (only used if load_into_memory=True)
        self._ngrams = None
        self._total_examples = None
        self._total_words = None

    def _load_ngrams(self):
        """Load all ngrams into memory with weighting applied."""
        if self._ngrams is not None:
            return  # Already loaded

        ngrams = []
        total_examples = 0
        total_words = 0

        ngram_stream = stream_year_ngrams(
            self.db_path,
            self.year,
            ngram_filter=self.ngram_filter
        )

        # Create progress bar if position is provided (parallel loading)
        pbar = None
        if self.progress_position is not None:
            pbar = tqdm(
                desc=f"Loading year {self.year}",
                unit=" ngrams",
                position=self.progress_position,
                leave=False  # Don't leave the bar after completion
            )

        for ngram_bytes, occurrences, documents in ngram_stream:
            ngram_str = ngram_bytes.decode('utf-8', errors='replace')
            ngram_tokens = ngram_str.split()

            # Apply weighting strategy
            if self.weight_by == "freq":
                weight = calculate_weight(occurrences, base=self.log_base)
            elif self.weight_by == "doc_freq":
                weight = calculate_weight(documents, base=self.log_base)
            else:
                weight = 1

            # Add weighted copies
            for _ in range(weight):
                ngrams.append(ngram_tokens)

            total_examples += weight
            total_words += len(ngram_tokens) * weight

            # Update progress bar
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self._ngrams = ngrams
        self._total_examples = total_examples
        self._total_words = total_words

    def __iter__(self):
        """
        Iterate over ngrams.

        Three modes:
        1. Stream from DB (load_into_memory=False): Consecutive repetition of weighted ngrams
        2. Load without shuffle (load_into_memory=True, shuffle=False): Fixed order, no I/O after load
        3. Load and shuffle (load_into_memory=True, shuffle=True): Randomized order each epoch
        """
        if self.load_into_memory:
            # Load data into memory if not already loaded
            self._load_ngrams()

            if self.shuffle:
                # Shuffle before each epoch
                shuffled = self._ngrams.copy()  # Shallow copy (just pointers)
                random.seed(self.random_seed)
                random.shuffle(shuffled)
                # No progress bar during training epochs - gensim shows its own
                for ngram_tokens in shuffled:
                    yield ngram_tokens
            else:
                # Iterate in original loaded order
                # No progress bar during training epochs - gensim shows its own
                for ngram_tokens in self._ngrams:
                    yield ngram_tokens
        else:
            # Streaming mode (original behavior)
            ngram_stream = stream_year_ngrams(
                self.db_path,
                self.year,
                ngram_filter=self.ngram_filter
            )

            # No progress bar during streaming - gensim shows its own training progress
            for ngram_bytes, occurrences, documents in ngram_stream:
                ngram_str = ngram_bytes.decode('utf-8', errors='replace')
                ngram_tokens = ngram_str.split()

                # Apply weighting strategy
                if self.weight_by == "freq":
                    weight = calculate_weight(occurrences, base=self.log_base)
                    yield from repeat(ngram_tokens, weight)
                elif self.weight_by == "doc_freq":
                    weight = calculate_weight(documents, base=self.log_base)
                    yield from repeat(ngram_tokens, weight)
                else:
                    yield ngram_tokens

    @property
    def total_examples(self):
        """Get total number of examples (for Word2Vec training)."""
        if self.load_into_memory:
            if self._ngrams is None:
                self._load_ngrams()
            return self._total_examples
        else:
            return None  # Not available in streaming mode

    @property
    def total_words(self):
        """Get total number of words (for Word2Vec training)."""
        if self.load_into_memory:
            if self._ngrams is None:
                self._load_ngrams()
            return self._total_words
        else:
            return None  # Not available in streaming mode


def configure_logging(log_dir, filename):
    """
    Configure and return a logger for a child process, adding Gensim's logs.

    Args:
        log_dir (str): Directory to store log files.
        filename (str): Name of the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, filename)
    logger_name = os.path.splitext(filename)[0]

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Attach to gensim logger
    gensim_logger = logging.getLogger("gensim")
    gensim_logger.handlers.clear()
    gensim_logger.setLevel(logging.INFO)
    gensim_logger.addHandler(file_handler)

    return logger


def train_word2vec(
        db_path,
        year,
        weight_by,
        vector_size,
        window,
        min_count,
        sg,
        workers,
        epochs,
        allow_unk=True,
        max_unk_count=None,
        load_into_memory=False,
        shuffle=False,
        random_seed=42,
        progress_position=None,
        **kwargs
):
    """
    Train a Word2Vec model on the given sentences from RocksDB.

    Args:
        db_path (str): Path to the pivoted RocksDB containing ngrams.
        year (int): Year to train on.
        weight_by (str): Weighting strategy ("freq", "doc_freq", or "none").
        vector_size (int): Size of word vectors.
        window (int): Context window size.
        min_count (int): Minimum frequency of words to include.
        sg (int): Training algorithm (1=skip-gram, 0=CBOW).
        workers (int): Number of worker threads.
        epochs (int): Number of training epochs.
        allow_unk (bool): Whether to allow ngrams with <UNK> tokens.
        max_unk_count (int): Maximum number of <UNK> tokens per ngram.
        load_into_memory (bool): If True, load ngrams into memory before training.
        shuffle (bool): If True, shuffle ngrams before each epoch (requires load_into_memory=True).
        random_seed (int): Random seed for shuffling.
        progress_position (int): Position for tqdm progress bar (for parallel loading).

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    sentences = SentencesIterable(
        db_path, year, weight_by=weight_by, log_base=10,
        allow_unk=allow_unk, max_unk_count=max_unk_count,
        load_into_memory=load_into_memory, shuffle=shuffle, random_seed=random_seed,
        progress_position=progress_position
    )

    if load_into_memory:
        # Build vocabulary separately for better control
        model = Word2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            workers=workers,
            **kwargs
        )

        model.build_vocab(sentences)

        # Note: Data is now loaded in memory (triggered by build_vocab iteration)
        # Vocab stats logged by gensim

        # Train with proper corpus statistics
        model.train(
            sentences,
            total_examples=sentences.total_examples,
            total_words=sentences.total_words,
            epochs=epochs
        )

        return model
    else:
        # Streaming mode: use original single-call approach
        return Word2Vec(
            sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            workers=workers,
            epochs=epochs,
            **kwargs
        )


def train_model(year, db_path, model_dir, log_dir, weight_by, vector_size,
                window, min_count, approach, epochs, workers, allow_unk=True,
                max_unk_count=None, load_into_memory=False, shuffle=False,
                random_seed=42, progress_position=None):
    """
    Train a Word2Vec model for a specific year from RocksDB.

    Args:
        year (int): Year to train on.
        db_path (str): Path to the pivoted RocksDB.
        model_dir (str): Directory to save trained models.
        log_dir (str): Directory to save log files.
        weight_by (str): Weighting strategy ("freq", "doc_freq", or "none").
        vector_size (int): Size of word vectors.
        window (int): Context window size.
        min_count (int): Minimum frequency of words to include.
        approach (str): Training approach ("skip-gram" or "CBOW").
        epochs (int): Number of training epochs.
        workers (int): Number of worker threads.
        allow_unk (bool): Whether to allow ngrams with <UNK> tokens.
        max_unk_count (int): Maximum number of <UNK> tokens per ngram.
        load_into_memory (bool): If True, load ngrams into memory before training.
        shuffle (bool): If True, shuffle ngrams before each epoch.
        random_seed (int): Random seed for shuffling.
        progress_position (int): Position for tqdm progress bar (for parallel loading).
    """
    # Set process title for monitoring
    if setproctitle is not None:
        try:
            setproctitle(f"ngt:y{year}_vs{vector_size}_w{window}")
        except Exception:
            pass  # Silently continue if setproctitle fails

    sg = 1 if approach == 'skip-gram' else 0

    name_string = (
        f"y{year}_wb{weight_by}_vs{vector_size}_w{window}_"
        f"mc{min_count}_sg{sg}_e{epochs}"
    )

    logger = configure_logging(
        log_dir,
        filename=f"w2v_{name_string}.log"
    )

    # Check if database exists
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        return

    os.makedirs(model_dir, exist_ok=True)

    try:
        logger.info(
            f"Processing year {year} with parameters: "
            f"vector_size={vector_size}, window={window}, "
            f"min_count={min_count}, sg={sg}, epochs={epochs}, "
            f"allow_unk={allow_unk}, max_unk_count={max_unk_count}, "
            f"load_into_memory={load_into_memory}, shuffle={shuffle}..."
        )

        if load_into_memory:
            logger.info(f"Loading year {year} into memory...")

        model = train_word2vec(
            db_path=db_path,
            year=year,
            weight_by=weight_by,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            epochs=epochs,
            workers=workers,
            allow_unk=allow_unk,
            max_unk_count=max_unk_count,
            load_into_memory=load_into_memory,
            shuffle=shuffle,
            random_seed=random_seed,
            progress_position=progress_position
        )

        model_filename = f"w2v_{name_string}.kv"
        model_save_path = os.path.join(model_dir, model_filename)
        model.wv.save(model_save_path)

        logger.info(f"Model for year {year} saved to {model_save_path}.")
    except Exception as e:
        logger.error(f"Error training model for year {year}: {e}", exc_info=True)


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
        print(f"No dir_suffix provided. Using timestamp: {dir_suffix}")

    weight_by = ensure_iterable(weight_by)
    vector_size = ensure_iterable(vector_size)
    window = ensure_iterable(window)
    min_count = ensure_iterable(min_count)
    approach = ensure_iterable(approach)
    epochs = ensure_iterable(epochs)

    start_time, db_path, model_dir, log_dir, ngram_size = set_info(
        corpus_path, dir_suffix
    )

    grid_params = (
        f'  Years:               {years[0]}–{years[1]} ({years[1] - years[0] + 1} years)\n'
        f'  Weighting:           {weight_by}\n'
        f'  Vector size:         {vector_size}\n'
        f'  Context window:      {window}\n'
        f'  Minimum word count:  {min_count}\n'
        f'  Approach:            {approach}\n'
        f'  Training epochs:     {epochs}\n'
        f'  Allow <UNK>:         {allow_unk}\n'
        f'  Max <UNK> count:     {max_unk_count}\n'
        f'  Load into memory:    {load_into_memory}\n'
        f'  Shuffle:             {shuffle}\n'
        f'  Parallel models:     {max_parallel_models}\n'
        f'  Workers per model:   {workers_per_model}'
    )

    print_info(
        start_time,
        db_path,
        model_dir,
        log_dir,
        ngram_size,
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

    print(f"Total tasks to run: {len(tasks)}")
    print(f"Running {max_parallel_models} models in parallel\n")

    # Use as_completed to dynamically manage position slots
    with ProcessPoolExecutor(max_workers=max_parallel_models) as executor:
        # Track available position slots
        available_positions = list(range(max_parallel_models))
        future_to_position = {}
        pending_tasks = list(tasks)

        # Submit initial batch
        for _ in range(min(max_parallel_models, len(pending_tasks))):
            if available_positions and pending_tasks:
                position = available_positions.pop(0)
                task = pending_tasks.pop(0)
                task_with_position = task + (position,)
                future = executor.submit(train_model, *task_with_position)
                future_to_position[future] = position

        # Monitor completion and submit new tasks
        completed = 0
        with tqdm(total=len(tasks), desc="Training Models", position=max_parallel_models, leave=True) as pbar:
            for future in as_completed(future_to_position):
                try:
                    future.result()
                except Exception as e:
                    print(f"Task failed with error: {e}")

                # Free up the position
                position = future_to_position.pop(future)
                available_positions.append(position)

                # Submit next task if any remain
                if pending_tasks:
                    position = available_positions.pop(0)
                    task = pending_tasks.pop(0)
                    task_with_position = task + (position,)
                    new_future = executor.submit(train_model, *task_with_position)
                    future_to_position[new_future] = position

                completed += 1
                pbar.update(1)