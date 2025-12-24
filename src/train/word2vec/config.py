"""Configuration and path utilities for Word2Vec training."""

import os
from datetime import datetime

__all__ = [
    "ensure_iterable",
    "construct_model_path",
    "set_info",
]


def ensure_iterable(param):
    """
    Ensure the input parameter is iterable (e.g., a tuple).

    Args:
        param: Parameter to check

    Returns:
        Tuple or list version of param
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


def set_info(corpus_path, dir_suffix, genre_focus=None):
    """
    Set up project paths for database, models, and logs.

    Args:
        corpus_path (str): Full path to corpus directory containing the database.
                          For ngrams: '/scratch/edk202/NLP_corpora/Google_Books/20200217/eng/5gram_files'
                          For Davies: '/scratch/edk202/NLP_corpora/COHA'
        dir_suffix (str): Suffix for model and log directories.
        genre_focus (list): Optional list of genres for Davies corpora (e.g., ['fic']).

    Returns:
        tuple: (start_time, db_path, model_dir, log_dir)
    """
    start_time = datetime.now()

    # Remove trailing slash for consistent path handling
    corpus_path = corpus_path.rstrip('/')

    # Check if this is an ngram corpus or a Davies corpus
    basename = os.path.basename(corpus_path)
    if 'gram_files' in basename:
        # Ngram corpus: Extract ngram size from corpus path (e.g., '5gram_files' -> 5)
        ngram_size = basename.replace('gram_files', '')
        db_path = os.path.join(corpus_path, f"{ngram_size}grams_pivoted.db")
        model_base = construct_model_path(corpus_path)
    else:
        # Davies corpus: Build database name based on genre_focus
        corpus_name = basename

        # Determine database name and genre-specific subdirectory
        if genre_focus is not None:
            # Build database name with genre suffix (matching ingestion/filtering)
            genre_suffix = "+".join(sorted(genre_focus))
            db_name = f"{corpus_name}_{genre_suffix}_filtered"
            # Create genre-specific subdirectory in model path
            genre_subdir = f"{corpus_name}_{genre_suffix}"
        else:
            # Plain corpus name (no genre filtering)
            db_name = f"{corpus_name}_filtered"
            # Use corpus_corpus pattern for consistency (e.g., COHA/COHA)
            genre_subdir = corpus_name

        db_path = os.path.join(corpus_path, db_name)

        if not os.path.exists(db_path):
            raise ValueError(
                f"Database not found at {db_path}. "
                f"Make sure you ran filter_davies_corpus() with genre_focus={genre_focus} first."
            )

        # Construct parallel model path with subdirectory (always includes subdirectory for consistency)
        model_base = construct_model_path(corpus_path)
        model_base = os.path.join(model_base, genre_subdir)

    # Model directory
    model_dir = os.path.join(model_base, f"models_{dir_suffix}")

    # Log directory
    log_dir = os.path.join(model_base, f"logs_{dir_suffix}", "training")

    return start_time, db_path, model_dir, log_dir
