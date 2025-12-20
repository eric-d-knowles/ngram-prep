import argparse
import os
import sys
import re
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool

from tqdm import tqdm
from ngramprep.common.w2v_model import W2VModel


def get_model_paths(model_dir):
    """
    Retrieve paths of all Word2Vec model files in the specified directory.
    Extracts the year robustly using regex.
    """
    model_paths = []
    pattern = re.compile(r'w2v_y(\d{4})')

    for f in Path(model_dir).glob("w2v_y*.kv"):
        match = pattern.search(f.name)
        if match:
            year = int(match.group(1))
            model_paths.append((year, str(f)))
        else:
            print(f"Skipping file with unexpected format: {f.name}")

    return sorted(model_paths)


def process_model(args):
    """
    Normalize and align a given model to the anchor model.
    """
    year, model_path, anchor_model, dir_suffix = args
    model = W2VModel(model_path)

    # Ensure vectors are writeable before normalization
    model.model.vectors = model.model.vectors.copy()
    model = model.normalize()

    if year != anchor_model[0]:
        model.filter_vocab(anchor_model[1].filtered_vocab)
        model.align_to(anchor_model[1])

    output_path = model_path.replace(f"models_{dir_suffix}",
                                     f"models_{dir_suffix}/norm_and_align")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)


def normalize_and_align_vectors(proj_dir=None, dir_suffix=None, anchor_year=None,
                                ngram_size=None, workers=None, corpus_path=None,
                                genre_focus=None):
    """
    Normalize and align Word2Vec models in the given project directory.

    Can be called in three ways:
    1. Explicit path mode: Provide proj_dir directly
    2. Auto-detect mode (Davies): Provide corpus_path, dir_suffix (and optionally genre_focus)
    3. Auto-detect mode (ngrams): Provide corpus_path with ngram_size

    Args:
        proj_dir: Project directory path (will be auto-derived if corpus_path provided)
        dir_suffix: Directory suffix (e.g., 'final', 'test')
        anchor_year: Year to use as anchor for alignment
        ngram_size: Optional. If provided, uses ngram structure (e.g., '5gram_files/models_{suffix}').
                    If None, uses flat structure (e.g., 'models_{suffix}')
        workers: Number of parallel workers (defaults to CPU count)
        corpus_path: Path to corpus directory (e.g., '/scratch/edk202/NLP_corpora/COHA') - used for auto-detection
        genre_focus: List of genres for Davies corpora (e.g., ['fic']) - used for Davies auto-detection

    Example:
        >>> # Auto-detect mode (Davies) - NEW PREFERRED METHOD
        >>> normalize_and_align_vectors(
        ...     corpus_path='/scratch/edk202/NLP_corpora/COHA',
        ...     dir_suffix='final',
        ...     anchor_year=2000,
        ...     workers=50
        ... )
        >>>
        >>> # Explicit path mode (old method, still works)
        >>> normalize_and_align_vectors(
        ...     proj_dir='/scratch/edk202/NLP_models/COHA',
        ...     dir_suffix='final',
        ...     anchor_year=2000,
        ...     workers=50
        ... )
    """
    # Set default workers
    if workers is None:
        workers = os.cpu_count()

    # Auto-derive proj_dir if corpus_path provided
    if proj_dir is None:
        if corpus_path is None:
            raise ValueError(
                "Either proj_dir or corpus_path must be provided. "
                "For Davies corpora, use: corpus_path='/path/to/COHA', dir_suffix='final'"
            )
        from .config import construct_model_path
        corpus_path = corpus_path.rstrip('/')
        proj_dir = construct_model_path(corpus_path)

    # Validate required parameters
    if dir_suffix is None:
        raise ValueError("dir_suffix parameter is required")
    if anchor_year is None:
        raise ValueError("anchor_year parameter is required")

    start_time = datetime.now()

    # Construct model directory based on whether ngram_size is provided
    if ngram_size is not None:
        model_dir = os.path.join(proj_dir, f'{ngram_size}gram_files/models_{dir_suffix}')
    else:
        model_dir = os.path.join(proj_dir, f'models_{dir_suffix}')

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_paths = get_model_paths(model_dir)
    if not model_paths:
        raise FileNotFoundError(f"No .kv models found in {model_dir}")

    # Load the anchor model
    anchor_model_path = next((p for y, p in model_paths if y == anchor_year), None)
    if not anchor_model_path:
        raise ValueError(f"Anchor model for year {anchor_year} not found.")

    anchor_model = W2VModel(anchor_model_path)
    anchor_model.model.vectors = anchor_model.model.vectors.copy()
    anchor_model = anchor_model.normalize()

    # Ensure anchor model has filtered_vocab before multiprocessing
    anchor_model.filter_vocab(anchor_model.extract_vocab())

    # Save the anchor model in the output directory
    output_anchor_path = anchor_model_path.replace(f"models_{dir_suffix}", f"models_{dir_suffix}/norm_and_align")
    Path(output_anchor_path).parent.mkdir(parents=True, exist_ok=True)
    anchor_model.save(output_anchor_path)  # Save normalized anchor model

    print(f"Saved normalized anchor model to {output_anchor_path}")

    # Prepare non-anchor models for multiprocessing
    tasks = [(y, p, (anchor_year, anchor_model), dir_suffix) for y, p in model_paths if y != anchor_year]

    with Pool(processes=workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_model, tasks),
            total=len(tasks),
            desc="Processing models",
            unit="file"
        ):
            pass

    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time}")
    print(f"Processed {len(model_paths)} models. Aligned to anchor year {anchor_year}.")
