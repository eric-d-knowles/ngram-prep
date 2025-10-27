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


def set_info(corpus_path, dir_suffix):
    """
    Set up project paths for database, models, and logs.

    Args:
        corpus_path (str): Full path to corpus directory containing the database.
                          e.g., '/scratch/edk202/NLP_corpora/Google_Books/20200217/eng/5gram_files'
        dir_suffix (str): Suffix for model and log directories.

    Returns:
        tuple: (start_time, db_path, model_dir, log_dir)
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

    return start_time, db_path, model_dir, log_dir
