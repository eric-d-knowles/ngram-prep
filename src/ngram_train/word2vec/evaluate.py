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
from tqdm import tqdm

from ngram_prep.utilities.display import truncate_path_to_fit
from .w2v_model import W2VModel

__all__ = ["evaluate_models"]

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


def _evaluate_a_model(model_path, similarity_dataset, analogy_dataset, model_logger):
    """
    Run intrinsic evaluations on a Word2Vec model.

    Args:
        model_path (str): Path to .kv model file
        similarity_dataset (str): Path to similarity evaluation dataset
        analogy_dataset (str): Path to analogy evaluation dataset
        model_logger (logging.Logger): Logger for this specific model

    Returns:
        dict: Dictionary with 'similarity_score' and 'analogy_score', or None on error
    """
    model_logger.info(f"Loading model from: {model_path}")

    try:
        model = W2VModel(model_path)

        similarity_score = model.evaluate("similarity", similarity_dataset)
        model_logger.info(f"Similarity Score (Spearman): {similarity_score}")

        analogy_score = model.evaluate("analogy", analogy_dataset)
        model_logger.info(f"Analogy Score: {analogy_score}")

        return {
            "similarity_score": similarity_score,
            "analogy_score": analogy_score
        }
    except Exception as e:
        model_logger.error(f"Error during evaluation: {e}")
        return None


def _evaluate_one_file(params):
    """
    Helper to evaluate a single model file with its own log file.

    Args:
        params (tuple): (file_name, model_dir, similarity_dataset, analogy_dataset, log_dir)

    Returns:
        dict or None: Evaluation results dictionary or None if evaluation failed
    """
    (file_name, model_dir, similarity_dataset, analogy_dataset, log_dir) = params

    metadata = _extract_model_metadata(file_name)
    if not metadata:
        # The filename doesn't match the pattern, skip
        return None

    (year, weight_by, vector_size, window,
     min_count, approach, epochs) = metadata

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
            model_logger=model_logger
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
            "similarity_score": evaluation["similarity_score"],
            "analogy_score": evaluation["analogy_score"]
        }
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
    workers
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
    """
    # Identify which files haven't been evaluated yet
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
        (f, model_dir, similarity_dataset, analogy_dataset, log_dir)
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
    model_dir,
    dir_suffix,
    eval_dir,
    save_mode='append',
    similarity_dataset=None,
    analogy_dataset=None,
    workers=None
):
    """
    Evaluate all Word2Vec models in a directory on similarity and analogy tasks.

    Args:
        model_dir (str): Base directory for models (will look in models_{dir_suffix} subdirectory).
        dir_suffix (str): Suffix used during training (for constructing paths and naming output files).
        eval_dir (str): Directory to save evaluation results CSV.
        save_mode (str): 'append' to add to existing results, 'overwrite' to replace.
                        Default: 'append'.
        similarity_dataset (str, optional): Path to similarity evaluation dataset.
                                           Defaults to gensim's wordsim353.
        analogy_dataset (str, optional): Path to analogy evaluation dataset.
                                        Defaults to gensim's questions-words.
        workers (int, optional): Number of parallel workers. Defaults to CPU count.

    Example:
        >>> from ngram_train.word2vec import evaluate_models
        >>> evaluate_models(
        ...     model_dir='/scratch/edk202/NLP_models/Google_Books/20200217/eng/5gram_files',
        ...     dir_suffix='test',
        ...     eval_dir='/scratch/edk202/NLP_models/Google_Books/20200217/eng/5gram_files',
        ...     save_mode='append',
        ...     workers=8
        ... )
    """
    if workers is None:
        workers = os.cpu_count()

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
        workers=workers
    )
