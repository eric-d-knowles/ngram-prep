"""
Batch normalization and alignment of Word2Vec models.

Provides functionality to normalize and align multiple year models to an anchor
year using orthogonal Procrustes transforms, with parallel processing support.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from .w2v_model import W2VModel
from .display import LINE_WIDTH
from ngramprep.utilities.display import truncate_path_to_fit

__all__ = ["normalize_and_align_models"]


def _normalize_model(model_path, output_dir):
    """
    Normalize a single model (worker function).

    Args:
        model_path (str): Path to the .kv model file.
        output_dir (str): Directory to save the normalized model.

    Returns:
        str: Path to the saved normalized model.
    """
    model = W2VModel(model_path)
    model.normalize()

    output_path = os.path.join(output_dir, os.path.basename(model_path))
    model.model.save(output_path)

    return output_path


def _align_model(model_path, anchor_path, shared_vocab, output_dir):
    """
    Align a single model to the anchor (worker function).

    Args:
        model_path (str): Path to the normalized .kv model file.
        anchor_path (str): Path to the anchor .kv model file.
        shared_vocab (set): Shared vocabulary across all models.
        output_dir (str): Directory to save the aligned model.

    Returns:
        str: Path to the saved aligned model.
    """
    model = W2VModel(model_path)
    anchor = W2VModel(anchor_path)

    model.filter_vocab(shared_vocab)
    anchor.filter_vocab(shared_vocab)

    model.align_to(anchor)

    output_path = os.path.join(output_dir, os.path.basename(model_path))
    model.save(output_path)

    return output_path


def normalize_and_align_models(
    model_dir,
    anchor_year,
    output_subdir="normalized_aligned",
    max_workers=None
):
    """
    Normalize and align multiple year models to an anchor year.

    This function performs a complete normalization and alignment pipeline:
    1. Loads all .kv models from the specified directory
    2. Normalizes all models in parallel
    3. Computes shared vocabulary across all models
    4. Aligns all models to the anchor year using Procrustes
    5. Saves normalized and aligned models to a subdirectory

    Args:
        model_dir (str): Directory containing .kv model files.
        anchor_year (str): Year to use as the alignment anchor (e.g., "1990").
        output_subdir (str): Subdirectory name for output models. Defaults to "normalized_aligned".
        max_workers (int, optional): Maximum number of parallel workers. Defaults to CPU count.

    Returns:
        str: Path to the output directory containing normalized and aligned models.

    Raises:
        FileNotFoundError: If model_dir doesn't exist or no .kv files are found.
        ValueError: If anchor_year model is not found.

    Example:
        >>> output_dir = normalize_and_align_models(
        ...     model_dir="/path/to/models",
        ...     anchor_year="1990",
        ...     max_workers=4
        ... )
    """
    start_time = datetime.now()
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Find all .kv files
    model_paths = sorted(model_dir.glob("*.kv"))
    if not model_paths:
        raise FileNotFoundError(f"No .kv files found in {model_dir}")

    # Find anchor model
    anchor_filename = f"{anchor_year}.kv"
    anchor_path = model_dir / anchor_filename
    if not anchor_path.exists():
        raise ValueError(f"Anchor model not found: {anchor_path}")

    # Create output directory
    output_dir = model_dir / output_subdir
    output_dir.mkdir(exist_ok=True)

    # Print header
    model_dir_str = truncate_path_to_fit(str(model_dir), "Model directory:      ", LINE_WIDTH)
    output_dir_str = truncate_path_to_fit(str(output_dir), "Output directory:     ", LINE_WIDTH)

    lines = [
        "",
        "WORD2VEC MODEL NORMALIZATION AND ALIGNMENT",
        "━" * LINE_WIDTH,
        f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}",
        "",
        "Configuration",
        "═" * LINE_WIDTH,
        f"Model directory:      {model_dir_str}",
        f"Output directory:     {output_dir_str}",
        f"Anchor year:          {anchor_year}",
        f"Total models:         {len(model_paths)}",
        f"Parallel workers:     {max_workers or 'auto'}",
        "",
    ]
    print("\n".join(lines), flush=True)

    # Step 1: Normalize all models
    print("Step 1: Normalizing models")
    print("─" * LINE_WIDTH)

    normalized_dir = output_dir / "temp_normalized"
    normalized_dir.mkdir(exist_ok=True)

    with tqdm(total=len(model_paths), desc="Normalizing", unit=" models") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_normalize_model, str(path), str(normalized_dir))
                for path in model_paths
            ]
            normalized_paths = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    normalized_paths.append(result)
                except Exception as e:
                    print(f"\nNormalization failed: {e}")
                pbar.update(1)

    print(f"✓ Normalized {len(normalized_paths)} models\n")

    # Step 2: Compute shared vocabulary
    print("Step 2: Computing shared vocabulary")
    print("─" * LINE_WIDTH)

    all_vocabs = []
    for path in tqdm(normalized_paths, desc="Loading vocabularies", unit=" models"):
        model = W2VModel(path)
        all_vocabs.append(model.extract_vocab())

    shared_vocab = set.intersection(*all_vocabs)
    print(f"✓ Shared vocabulary size: {len(shared_vocab):,}\n")

    # Step 3: Align all models to anchor
    print("Step 3: Aligning models to anchor")
    print("─" * LINE_WIDTH)

    normalized_anchor = normalized_dir / anchor_filename

    with tqdm(total=len(normalized_paths), desc="Aligning", unit=" models") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _align_model,
                    path,
                    str(normalized_anchor),
                    shared_vocab,
                    str(output_dir)
                )
                for path in normalized_paths
            ]
            aligned_paths = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    aligned_paths.append(result)
                except Exception as e:
                    print(f"\nAlignment failed: {e}")
                pbar.update(1)

    print(f"✓ Aligned {len(aligned_paths)} models\n")

    # Clean up temporary directory
    import shutil
    shutil.rmtree(normalized_dir)

    # Print completion banner
    end_time = datetime.now()
    duration = end_time - start_time

    lines = [
        "Normalization and Alignment Complete",
        "═" * LINE_WIDTH,
        f"Models processed:     {len(aligned_paths)}",
        f"Shared vocab size:    {len(shared_vocab):,}",
        f"Output directory:     {output_dir_str}",
        f"Duration:             {duration}",
        "━" * LINE_WIDTH,
        "",
    ]
    print("\n".join(lines), flush=True)

    return str(output_dir)
