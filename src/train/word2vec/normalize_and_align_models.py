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
    year, model_path, anchor_model, dir_suffix, stability_weights = args
    model = W2VModel(model_path)

    # Ensure vectors are writeable before normalization
    model.model.vectors = model.model.vectors.copy()
    model = model.normalize()

    if year != anchor_model[0]:
        model.filter_vocab(anchor_model[1].filtered_vocab)
        model.align_to(anchor_model[1], weights=stability_weights)

    output_path = model_path.replace(f"models_{dir_suffix}",
                                     f"models_{dir_suffix}/norm_and_align")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)


def normalize_and_align_vectors(proj_dir=None, dir_suffix=None, anchor_year=None,
                                ngram_size=None, workers=None, corpus_path=None,
                                genre_focus=None, weighted_alignment=False,
                                stability_method='local_stability',
                                include_frequency=True, frequency_weight=0.3):
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
        weighted_alignment: If True, use stability-weighted Procrustes (more stable words have more influence).
                           If False (default), use unweighted Procrustes (all words contribute equally).
        stability_method: Method for computing stability weights ('local_stability', 'global_stability',
                         'frequency_stability', 'combined'). Only used if weighted_alignment=True
        include_frequency: If True, incorporate word frequency into weights (recommended).
                          More frequent words have more reliable embeddings. Only used if weighted_alignment=True
        frequency_weight: Weight for frequency component (0.0-1.0). Default 0.3 gives 70% weight to stability,
                         30% to frequency. Only used if weighted_alignment=True and include_frequency=True

    Example:
        >>> # Basic usage: unweighted alignment (default)
        >>> normalize_and_align_vectors(
        ...     corpus_path='/scratch/edk202/NLP_corpora/COHA',
        ...     dir_suffix='final',
        ...     anchor_year=2000,
        ...     workers=50
        ... )
        >>>
        >>> # Stability-weighted alignment with frequency (recommended)
        >>> normalize_and_align_vectors(
        ...     corpus_path='/scratch/edk202/NLP_corpora/COHA',
        ...     dir_suffix='final',
        ...     anchor_year=2000,
        ...     weighted_alignment=True,
        ...     stability_method='local_stability',
        ...     include_frequency=True,
        ...     frequency_weight=0.3,
        ...     workers=50
        ... )
        >>>
        >>> # Using combined stability metric with custom frequency weight
        >>> normalize_and_align_vectors(
        ...     corpus_path='/scratch/edk202/NLP_corpora/COHA',
        ...     dir_suffix='final',
        ...     anchor_year=2000,
        ...     weighted_alignment=True,
        ...     stability_method='combined',
        ...     include_frequency=True,
        ...     frequency_weight=0.4,
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

        # Add genre-specific subdirectory for Davies corpora
        corpus_name = os.path.basename(corpus_path)
        if genre_focus is not None:
            genre_suffix = "+".join(sorted(genre_focus))
            genre_subdir = f"{corpus_name}_{genre_suffix}"
        else:
            # Use corpus_corpus pattern for consistency (e.g., COHA/COHA)
            genre_subdir = corpus_name
        proj_dir = os.path.join(proj_dir, genre_subdir)

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

    # Determine output directory
    output_dir = model_dir.replace(f"models_{dir_suffix}", f"models_{dir_suffix}/norm_and_align")

    # Print header
    from .display import print_alignment_header
    print_alignment_header(
        start_time=start_time,
        model_dir=model_dir,
        output_dir=output_dir,
        anchor_year=anchor_year,
        num_models=len(model_paths),
        weighted_alignment=weighted_alignment,
        stability_method=stability_method if weighted_alignment else None,
        include_frequency=include_frequency if weighted_alignment else None,
        frequency_weight=frequency_weight if weighted_alignment else None,
        workers=workers
    )

    # Load the anchor model
    anchor_model_path = next((p for y, p in model_paths if y == anchor_year), None)
    if not anchor_model_path:
        raise ValueError(f"Anchor model for year {anchor_year} not found.")

    anchor_model = W2VModel(anchor_model_path)
    anchor_model.model.vectors = anchor_model.model.vectors.copy()
    anchor_model = anchor_model.normalize()

    # Ensure anchor model has filtered_vocab before multiprocessing
    anchor_model.filter_vocab(anchor_model.extract_vocab())

    # Compute stability weights if using weighted alignment
    stability_weights = None
    if weighted_alignment:
        from .stability_weighting import load_models_for_stability_weighting, compute_stability_weights

        print("")
        print("Stability Weight Computation")
        print("═" * 100)
        models_for_weighting, shared_vocab = load_models_for_stability_weighting(model_paths, verbose=True)

        stability_weights = compute_stability_weights(
            models=models_for_weighting,
            shared_vocab=shared_vocab,
            method=stability_method,
            include_frequency=include_frequency,
            frequency_weight=frequency_weight,
            verbose=True
        )
        print("")

    # Save the anchor model in the output directory
    output_anchor_path = anchor_model_path.replace(f"models_{dir_suffix}", f"models_{dir_suffix}/norm_and_align")
    Path(output_anchor_path).parent.mkdir(parents=True, exist_ok=True)
    anchor_model.save(output_anchor_path)

    # Prepare non-anchor models for multiprocessing
    tasks = [(y, p, (anchor_year, anchor_model), dir_suffix, stability_weights) for y, p in model_paths if y != anchor_year]

    print("Processing Models")
    print("═" * 100)
    with Pool(processes=workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_model, tasks),
            total=len(tasks),
            desc="Aligning models",
            unit=" models"
        ):
            pass

    # Print completion banner
    end_time = datetime.now()
    runtime = end_time - start_time

    from .display import print_alignment_completion
    print_alignment_completion(
        output_dir=output_dir,
        num_models=len(model_paths),
        runtime=runtime
    )
