"""
Intrinsic evaluation of Word2Vec models.

Evaluates trained models on similarity and analogy tasks,
with parallel processing and per-model logging.
"""

import logging
import os
import re
from datetime import datetime
from multiprocessing import Pool

import pandas as pd
from gensim.test.utils import datapath
from setproctitle import setproctitle
from tqdm import tqdm

from ngramprep.utilities.display import truncate_path_to_fit
from .w2v_model import W2VModel

__all__ = ["evaluate_models", "evaluate_word2vec_models"]

LINE_WIDTH = 100


def _set_info(model_dir, dir_suffix, eval_dir, similarity_dataset, analogy_dataset):
    """
    Set up evaluation paths and validate directories.

    Args:
        model_dir (str): Base directory for models (will be appended with models_{dir_suffix})
        dir_suffix (str): Suffix used during training (for output naming)
        eval_dir (str): Directory to save evaluation results CSV
        similarity_dataset (str): Path to similarity evaluation dataset
        analogy_dataset (str): Path to analogy evaluation dataset

    Returns:
        tuple: (start_time, model_dir, eval_file, log_dir, similarity_dataset, analogy_dataset)
               or None if validation fails
    """
    start_time = datetime.now()

    # Construct full model directory path
    model_dir = os.path.join(model_dir, f"models_{dir_suffix}")

    # Validate model directory
    if not os.path.exists(model_dir):
        print(f"Error: Model directory does not exist: {model_dir}")
        return None

    # Construct evaluation output file path
    if not os.path.exists(eval_dir):
        print(f"Error: Evaluation directory does not exist: {eval_dir}")
        return None

    eval_file = os.path.join(eval_dir, f"evaluation_results_{dir_suffix}.csv")

    # Construct logging path (create if necessary)
    log_dir = os.path.join(os.path.dirname(model_dir), f"logs_{dir_suffix}", "evaluation")
    os.makedirs(log_dir, exist_ok=True)

    # Get similarity and analogy datasets (use gensim defaults if not provided)
    if similarity_dataset is None:
        similarity_dataset = datapath('wordsim353.tsv')
    if analogy_dataset is None:
        analogy_dataset = datapath('questions-words.txt')

    return (
        start_time,
        model_dir,
        eval_file,
        log_dir,
        similarity_dataset,
        analogy_dataset
    )


def _print_info(
    start_time,
    model_dir,
    eval_file,
    log_dir,
    similarity_dataset,
    analogy_dataset,
    save_mode,
):
    """Print evaluation configuration header."""
    # Format paths using truncate_path_to_fit
    model_dir_str = truncate_path_to_fit(model_dir, "Model directory:      ", LINE_WIDTH)
    eval_file_str = truncate_path_to_fit(eval_file, "Evaluation file:      ", LINE_WIDTH)
    log_dir_str = truncate_path_to_fit(log_dir, "Log directory:        ", LINE_WIDTH)
    sim_dataset_str = truncate_path_to_fit(similarity_dataset, "Similarity dataset:   ", LINE_WIDTH)
    ana_dataset_str = truncate_path_to_fit(analogy_dataset, "Analogy dataset:      ", LINE_WIDTH)

    lines = [
        "",
        "WORD2VEC MODEL EVALUATION",
        "━" * LINE_WIDTH,
        f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}",
        "",
        "Configuration",
        "═" * LINE_WIDTH,
        f"Model directory:      {model_dir_str}",
        f"Evaluation file:      {eval_file_str}",
        f"Log directory:        {log_dir_str}",
        f"Save mode:            {save_mode}",
        "",
        "Evaluation Datasets",
        "─" * LINE_WIDTH,
        f"Similarity dataset:   {sim_dataset_str}",
        f"Analogy dataset:      {ana_dataset_str}",
        "",
    ]
    print("\n".join(lines), flush=True)


def _extract_model_metadata(file_name):
    """
    Extract metadata from the model filename using regex.

    Args:
        file_name (str): Model filename (e.g., 'w2v_y1800_wbnone_vs300_w2_mc1_sg1_e5.kv')

    Returns:
        tuple or None: (year, weight_by, vector_size, window, min_count, approach, epochs)
                       or None if pattern doesn't match
    """
    pattern = re.compile(
        r"w2v_y(\d+)_wb(\w+)_vs(\d+)_w(\d+)_mc(\d+)_sg(\d+)_e(\d+)\.kv"
    )
    match = pattern.match(file_name)
    if match:
        return match.groups()
    return None


def _evaluate_a_model(model_path, similarity_dataset, analogy_dataset, model_logger,
                      run_similarity=True, run_analogy=True):
    """
    Run intrinsic evaluations on a Word2Vec model.

    Args:
        model_path (str): Path to .kv model file
        similarity_dataset (str): Path to similarity evaluation dataset
        analogy_dataset (str): Path to analogy evaluation dataset
        model_logger (logging.Logger): Logger for this specific model
        run_similarity (bool): Whether to run similarity evaluation
        run_analogy (bool): Whether to run analogy evaluation

    Returns:
        dict: Dictionary with 'similarity_score' and/or 'analogy_score', or None on error
    """
    model_logger.info(f"Loading model from: {model_path}")

    try:
        model = W2VModel(model_path)
        results = {}

        if run_similarity:
            similarity_score = model.evaluate("similarity", similarity_dataset)
            model_logger.info(f"Similarity Score (Spearman): {similarity_score}")
            results["similarity_score"] = similarity_score

        if run_analogy:
            analogy_score = model.evaluate("analogy", analogy_dataset)
            model_logger.info(f"Analogy Score: {analogy_score}")
            results["analogy_score"] = analogy_score

        return results
    except Exception as e:
        model_logger.error(f"Error during evaluation: {e}")
        return None


def _evaluate_one_file(params):
    """
    Helper to evaluate a single model file with its own log file.

    Args:
        params (tuple): (file_name, model_dir, similarity_dataset, analogy_dataset, log_dir,
                        run_similarity, run_analogy)

    Returns:
        dict or None: Evaluation results dictionary or None if evaluation failed
    """
    (file_name, model_dir, similarity_dataset, analogy_dataset, log_dir,
     run_similarity, run_analogy) = params

    metadata = _extract_model_metadata(file_name)
    if not metadata:
        # The filename doesn't match the pattern, skip
        return None

    (year, weight_by, vector_size, window,
     min_count, sg_str, epochs) = metadata

    # Convert sg value to approach name
    sg = int(sg_str)
    approach = 'skip-gram' if sg == 1 else 'cbow'

    # Set process title with nge: prefix using full model naming pattern
    name_string = (
        f"y{year}_wb{weight_by}_vs{int(vector_size):03d}_w{int(window):03d}_"
        f"mc{int(min_count):03d}_sg{sg}_e{int(epochs):03d}"
    )
    setproctitle(f"nge:{name_string}")

    model_path = os.path.join(model_dir, file_name)

    # Create a unique log file for this model
    model_log_file = os.path.join(
        log_dir,
        f"{os.path.splitext(file_name)[0]}.log"
    )

    # Set up a local logger for this specific model
    model_logger = logging.getLogger(f"logger_{file_name}")
    model_logger.setLevel(logging.INFO)

    # Clear any existing handlers to avoid duplication
    while model_logger.handlers:
        model_logger.handlers.pop()

    file_handler = logging.FileHandler(model_log_file, mode='a')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    model_logger.addHandler(file_handler)

    # Start logging
    model_logger.info("--------------------------------------------------")
    model_logger.info(f"Beginning evaluation for model: {file_name}")

    try:
        evaluation = _evaluate_a_model(
            model_path,
            similarity_dataset=similarity_dataset,
            analogy_dataset=analogy_dataset,
            model_logger=model_logger,
            run_similarity=run_similarity,
            run_analogy=run_analogy
        )
        if not evaluation:
            model_logger.info("Evaluation returned None.")
            return None

        result_dict = {
            "model": file_name,
            "year": int(year),
            "weight_by": weight_by,
            "vector_size": int(vector_size),
            "window": int(window),
            "min_count": int(min_count),
            "approach": approach,
            "epochs": int(epochs),
        }

        if run_similarity:
            result_dict["similarity_score"] = evaluation["similarity_score"]
        if run_analogy:
            result_dict["analogy_score"] = evaluation["analogy_score"]
        model_logger.info(f"Evaluation completed for {file_name}")
        return result_dict

    except Exception as e:
        model_logger.error(f"Error evaluating {file_name}: {e}")
        return None

    finally:
        # Close the file handler so logs are flushed
        model_logger.removeHandler(file_handler)
        file_handler.close()


def _evaluate_models_in_directory(
    model_dir,
    eval_file,
    log_dir,
    save_mode,
    similarity_dataset,
    analogy_dataset,
    workers,
    run_similarity=True,
    run_analogy=True
):
    """
    Evaluate all Word2Vec models in a directory using multiprocessing.

    Args:
        model_dir (str): Directory containing .kv model files
        eval_file (str): Path to output CSV file
        log_dir (str): Directory for individual model log files
        save_mode (str): 'append' or 'overwrite'
        similarity_dataset (str): Path to similarity evaluation dataset
        analogy_dataset (str): Path to analogy evaluation dataset
        workers (int): Number of parallel workers
        run_similarity (bool): Whether to run similarity evaluation
        run_analogy (bool): Whether to run analogy evaluation
    """
    # Identify which files haven't been evaluated yet
    if save_mode == 'overwrite':
        # In overwrite mode, evaluate all models
        files_to_evaluate = [
            f for f in os.listdir(model_dir)
            if f.endswith('.kv')
        ]
    else:
        # In append mode, only evaluate models not in existing CSV
        if os.path.isfile(eval_file):
            existing = pd.read_csv(eval_file)['model'].values
        else:
            existing = []

        files_to_evaluate = [
            f for f in os.listdir(model_dir)
            if f.endswith('.kv') and f not in existing
        ]

    if not files_to_evaluate:
        print("No new models to evaluate.")
        return

    print(f"Found {len(files_to_evaluate)} models to evaluate")
    print("")

    # Build list of parameter tuples for parallel processing
    param_list = [
        (f, model_dir, similarity_dataset, analogy_dataset, log_dir,
         run_similarity, run_analogy)
        for f in files_to_evaluate
    ]

    results = []
    with Pool(processes=workers) as pool:
        # imap_unordered yields results as they come in
        for result in tqdm(
            pool.imap_unordered(_evaluate_one_file, param_list),
            total=len(param_list),
            desc="Evaluating models",
            unit=" models"
        ):
            if result is not None:
                results.append(result)

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        if save_mode == 'overwrite':
            df.to_csv(eval_file, mode='w', index=False)
        else:
            file_exists = os.path.isfile(eval_file)
            df.to_csv(eval_file, mode='a', index=False, header=not file_exists)

        print("")
        print("Evaluation Complete")
        print("═" * LINE_WIDTH)
        print(f"Models evaluated:     {len(results)}")
        print(f"Results saved to:     {truncate_path_to_fit(eval_file, 'Results saved to:     ', LINE_WIDTH)}")
        print("━" * LINE_WIDTH)
        print("")
    else:
        print("\n⚠️ Warning: No valid results were returned after evaluation.\n")


def evaluate_models(
    model_dir=None,
    dir_suffix=None,
    eval_dir=None,
    save_mode='append',
    similarity_dataset=None,
    analogy_dataset=None,
    workers=None,
    run_similarity=True,
    run_analogy=True,
    corpus_path=None,
    genre_focus=None
):
    """
    Evaluate all Word2Vec models in a directory on similarity and analogy tasks.

    Two usage modes:
    1. Explicit paths (backwards compatible):
       evaluate_models(model_dir='/path/to/models', dir_suffix='test', eval_dir='/path')

    2. Auto-derived paths (new, simpler):
       evaluate_models(corpus_path='/scratch/.../COHA', dir_suffix='test', genre_focus=['fic'])

    Args:
        model_dir (str, optional): Base directory for models (will look in models_{dir_suffix} subdirectory).
                                  If None, derived from corpus_path and genre_focus.
        dir_suffix (str): Suffix used during training (for constructing paths and naming output files).
        eval_dir (str, optional): Directory to save evaluation results CSV. If None, uses model_dir.
        save_mode (str): 'append' to add to existing results, 'overwrite' to replace.
                        Default: 'append'.
        similarity_dataset (str, optional): Path to similarity evaluation dataset.
                                           Defaults to gensim's wordsim353.
        analogy_dataset (str, optional): Path to analogy evaluation dataset.
                                        Defaults to gensim's questions-words.
        workers (int, optional): Number of parallel workers. Defaults to CPU count.
        run_similarity (bool): Whether to run similarity evaluation. Default: True.
        run_analogy (bool): Whether to run analogy evaluation. Default: True.
        corpus_path (str, optional): Path to corpus directory (e.g., '/scratch/.../COHA').
                                    Used to auto-derive model_dir if model_dir not provided.
        genre_focus (list, optional): List of genres for Davies corpora (e.g., ['fic']).

    Example:
        >>> # New simplified usage:
        >>> evaluate_models(
        ...     corpus_path='/scratch/edk202/NLP_corpora/COHA',
        ...     dir_suffix='test',
        ...     genre_focus=['fic'],
        ...     workers=8
        ... )
        >>>
        >>> # Old explicit usage still works:
        >>> evaluate_models(
        ...     model_dir='/scratch/edk202/NLP_models/Google_Books/.../5gram_files',
        ...     dir_suffix='test',
        ...     eval_dir='/scratch/edk202/NLP_models/Google_Books/.../5gram_files',
        ...     workers=8
        ... )
    """
    if workers is None:
        workers = os.cpu_count()

    # Auto-derive model_dir and eval_dir if corpus_path provided
    if model_dir is None:
        if corpus_path is None:
            raise ValueError(
                "Either model_dir or corpus_path must be provided. "
                "For Davies corpora, use: corpus_path='/path/to/COHA', genre_focus=['fic']"
            )
        # Use construct_model_path from config module
        from .config import construct_model_path
        corpus_path = corpus_path.rstrip('/')
        model_dir = construct_model_path(corpus_path)

    if eval_dir is None:
        eval_dir = model_dir

    info = _set_info(
        model_dir,
        dir_suffix,
        eval_dir,
        similarity_dataset,
        analogy_dataset
    )
    if not info:
        return

    (
        start_time,
        model_dir,
        eval_file,
        log_dir,
        similarity_dataset,
        analogy_dataset
    ) = info

    _print_info(
        start_time,
        model_dir,
        eval_file,
        log_dir,
        similarity_dataset,
        analogy_dataset,
        save_mode
    )

    _evaluate_models_in_directory(
        model_dir=model_dir,
        eval_file=eval_file,
        log_dir=log_dir,
        save_mode=save_mode,
        similarity_dataset=similarity_dataset,
        analogy_dataset=analogy_dataset,
        workers=workers,
        run_similarity=run_similarity,
        run_analogy=run_analogy
    )


def _setup_logging(log_dir):
    """
    Set up logging for the evaluation pipeline if not already configured.

    Args:
        log_dir: Directory to store log files
    """
    # Check if logging is already configured
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        # Logging already configured, don't override
        return

    # Set up logging to log directory
    from ngramprep.ngram_acquire.logger import setup_logger

    log_file = setup_logger(
        db_path=str(log_dir),
        filename_prefix="word2vec_evaluation",
        console=False,
        rotate=True,
        max_bytes=100_000_000,
        backup_count=5,
        force=False  # Don't override if already configured
    )

    # Log initial context
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Word2Vec Evaluation Pipeline")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)


def evaluate_word2vec_models(
    ngram_size=None,
    repo_release_id=None,
    repo_corpus_id=None,
    db_path_stub=None,
    dir_suffix=None,
    save_mode='append',
    similarity_dataset=None,
    analogy_dataset=None,
    workers=None,
    run_similarity=True,
    run_analogy=True
):
    """
    Convenience wrapper for evaluating Word2Vec models using path stub parameters.

    This function mirrors the interface of build_word2vec_models,
    taking simple path parameters instead of requiring manual path construction.

    Args:
        ngram_size (int): N-gram size (e.g., 5 for 5grams) - required
        repo_release_id (str): Release date in YYYYMMDD format (e.g., "20200217") - required
        repo_corpus_id (str): Corpus identifier (e.g., "eng", "eng-fiction") - required
        db_path_stub (str): Base directory for data (e.g., "/scratch/edk202/NLP_corpora/Google_Books/") - required
        dir_suffix (str): Suffix for model/log directories (e.g., 'test', 'final') - required
        save_mode (str): 'append' to add to existing results, 'overwrite' to replace (default: 'append')
        similarity_dataset (str, optional): Path to similarity evaluation dataset
        analogy_dataset (str, optional): Path to analogy evaluation dataset
        workers (int, optional): Number of parallel workers (default: CPU count)
        run_similarity (bool): Whether to run similarity evaluation (default: True)
        run_analogy (bool): Whether to run analogy evaluation (default: True)

    Returns:
        str: Path to the evaluation results CSV file

    Example:
        >>> evaluate_word2vec_models(
        ...     ngram_size=5,
        ...     repo_release_id='20200217',
        ...     repo_corpus_id='eng-fiction',
        ...     db_path_stub='/scratch/edk202/NLP_corpora/Google_Books/',
        ...     dir_suffix='final',
        ...     run_similarity=True,
        ...     run_analogy=True,
        ...     workers=100
        ... )
    """
    # Validate required parameters
    if ngram_size is None or repo_release_id is None or repo_corpus_id is None or db_path_stub is None:
        raise ValueError(
            "All path stub parameters are required: "
            "ngram_size, repo_release_id, repo_corpus_id, db_path_stub"
        )

    if dir_suffix is None:
        raise ValueError("dir_suffix parameter is required (e.g., 'test', 'final')")

    # Construct paths from stub parameters
    from ngramprep.ngram_acquire.db.build_path import build_db_path
    from pathlib import Path

    base_path = Path(build_db_path(db_path_stub, ngram_size, repo_release_id, repo_corpus_id)).parent

    # Construct model path (replaces NLP_corpora with NLP_models)
    from .config import construct_model_path
    model_base = construct_model_path(str(base_path))

    # Set up logging internally
    log_dir = os.path.join(model_base, f"logs_{dir_suffix}", "evaluation")
    os.makedirs(log_dir, exist_ok=True)
    _setup_logging(log_dir)

    # Call the main evaluation function
    evaluate_models(
        model_dir=model_base,
        dir_suffix=dir_suffix,
        eval_dir=model_base,
        save_mode=save_mode,
        similarity_dataset=similarity_dataset,
        analogy_dataset=analogy_dataset,
        workers=workers,
        run_similarity=run_similarity,
        run_analogy=run_analogy
    )

    # Return evaluation results file path
    eval_file = os.path.join(model_base, f"evaluation_results_{dir_suffix}.csv")
    return eval_file
