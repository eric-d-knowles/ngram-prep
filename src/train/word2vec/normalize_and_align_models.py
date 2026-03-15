import os
import re
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool

from tqdm import tqdm
from ngramprep.common.w2v_model import W2VModel
from .display import LINE_WIDTH


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


def make_output_path(model_path: str, dir_suffix: str) -> str:
    """
    Derive the output path for a normalized/aligned model by inserting
    'norm_and_align' as a subdirectory of the models directory.

    Uses Path surgery rather than string replacement to avoid incorrect
    substitution when dir_suffix appears elsewhere in the path.
    """
    p = Path(model_path)
    return str(p.parent / "norm_and_align" / p.name)


def process_model(args):
    """
    Normalize, vocab-filter, and (for non-anchor years) align a model to
    the anchor. The anchor year is also passed through this function so
    that all models — including the anchor — are saved via the same code
    path and receive identical normalization and vocab-filtering treatment.
    """
    year, model_path, anchor_year, anchor_model_path, dir_suffix, stability_weights = args

    # Load and normalize. W2VModel.normalize() is responsible for the
    # read-only copy internally; callers should not need to do it here.
    model = W2VModel(model_path)
    model = model.normalize()

    if year == anchor_year:
        # Anchor: filter to its own vocab to establish the shared vocabulary
        # baseline. All other models will be filtered to this same set.
        model.filter_vocab(model.extract_vocab())
    else:
        # Load the anchor from disk (cheap — .kv files are memory-mapped)
        # rather than pickling the full anchor object into every worker.
        anchor = W2VModel(anchor_model_path)
        anchor = anchor.normalize()
        anchor.filter_vocab(anchor.extract_vocab())

        model.filter_vocab(anchor.filtered_vocab)
        model.align_to(anchor, weights=stability_weights)

    output_path = make_output_path(model_path, dir_suffix)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)


def normalize_and_align_vectors(
        proj_dir=None,
        dir_suffix=None,
        anchor_year=None,
        ngram_size=None,
        workers=None,
        corpus_path=None,
        genre_focus=None,
        weighted_alignment=False,
        stability_method='local_stability',
        include_frequency=True,
        frequency_weight=0.3,
        repo_release_id=None,
        repo_corpus_id=None,
        db_path_stub=None
):
    """
    Normalize and align Word2Vec models in the given project directory.

    Can be called in four ways:
    1. Explicit path mode: Provide proj_dir directly
    2. Auto-detect mode (Davies): Provide corpus_path, dir_suffix (and optionally genre_focus)
    3. Auto-detect mode (ngrams with corpus_path): Provide corpus_path with ngram_size
    4. Auto-detect mode (ngrams with stubs): Provide ngram_size, repo_release_id, repo_corpus_id, db_path_stub

    Args:
        proj_dir: Project directory path (will be auto-derived if corpus_path or db_path_stub provided)
        dir_suffix: Directory suffix (e.g., 'final', 'test')
        anchor_year: Year to use as anchor for alignment
        ngram_size: N-gram size (e.g., 5 for 5grams). Required for ngram mode.
        workers: Number of parallel workers. Defaults to SLURM_CPUS_PER_TASK if
                 available, otherwise os.cpu_count().
        corpus_path: Path to corpus directory (e.g., '/scratch/edk202/NLP_corpora/COHA')
        genre_focus: List of genres for Davies corpora (e.g., ['fic'])
        weighted_alignment: If True, use stability-weighted Procrustes.
        stability_method: Method for computing stability weights ('local_stability',
                         'global_stability', 'frequency_stability', 'combined').
                         Only used if weighted_alignment=True.
        include_frequency: If True, incorporate word frequency into weights.
                          Only used if weighted_alignment=True.
        frequency_weight: Weight for frequency component (0.0-1.0). Default 0.3.
                         Only used if weighted_alignment=True and include_frequency=True.
        repo_release_id: Release date in YYYYMMDD format (e.g., "20200217")
        repo_corpus_id: Corpus identifier (e.g., "eng", "eng-fiction")
        db_path_stub: Base directory for data (e.g., "/scratch/edk202/NLP_corpora/Google_Books/")

    Example:
        >>> # Google Books ngram mode (using stub parameters):
        >>> normalize_and_align_vectors(
        ...     ngram_size=5,
        ...     repo_release_id='20200217',
        ...     repo_corpus_id='eng',
        ...     db_path_stub='/scratch/edk202/NLP_corpora/Google_Books/',
        ...     dir_suffix='final',
        ...     anchor_year=2000,
        ...     workers=50
        ... )
        >>>
        >>> # Davies corpus mode (COHA, COCA, etc.):
        >>> normalize_and_align_vectors(
        ...     corpus_path='/scratch/edk202/NLP_corpora/COHA',
        ...     dir_suffix='final',
        ...     anchor_year=2000,
        ...     genre_focus=['fic'],
        ...     workers=50
        ... )
        >>>
        >>> # Stability-weighted alignment with frequency (recommended):
        >>> normalize_and_align_vectors(
        ...     ngram_size=5,
        ...     repo_release_id='20200217',
        ...     repo_corpus_id='eng',
        ...     db_path_stub='/scratch/edk202/NLP_corpora/Google_Books/',
        ...     dir_suffix='final',
        ...     anchor_year=2000,
        ...     weighted_alignment=True,
        ...     stability_method='local_stability',
        ...     include_frequency=True,
        ...     frequency_weight=0.3,
        ...     workers=50
        ... )
        >>>
        >>> # Explicit path mode (backwards compatible):
        >>> normalize_and_align_vectors(
        ...     proj_dir='/scratch/edk202/NLP_models/Google_Books/20200217/eng/5gram_files',
        ...     dir_suffix='final',
        ...     anchor_year=2000,
        ...     workers=50
        ... )
    """
    # Default workers: respect SLURM allocation on HPC; fall back to CPU count
    if workers is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        workers = int(slurm_cpus) if slurm_cpus is not None else os.cpu_count()

    # --- Resolve proj_dir ---
    resolved_proj_dir = proj_dir  # preserve original argument, work on a local copy

    if resolved_proj_dir is None:
        if db_path_stub is not None:
            if ngram_size is None or repo_release_id is None or repo_corpus_id is None:
                raise ValueError(
                    "When using db_path_stub, all ngram parameters are required: "
                    "ngram_size, repo_release_id, repo_corpus_id, db_path_stub"
                )
            from ngramprep.ngram_acquire.db.build_path import build_db_path

            db_full_path = build_db_path(
                db_path_stub.rstrip('/'), ngram_size, repo_release_id, repo_corpus_id
            )
            resolved_proj_dir = str(Path(db_full_path).parent).replace(
                'NLP_corpora', 'NLP_models'
            )

        elif corpus_path is not None:
            from .config import construct_model_path

            corpus_path = corpus_path.rstrip('/')
            resolved_proj_dir = construct_model_path(corpus_path)

            corpus_name = os.path.basename(corpus_path)
            genre_subdir = (
                f"{corpus_name}_{''.join(sorted(genre_focus))}"
                if genre_focus is not None
                else corpus_name
            )
            resolved_proj_dir = os.path.join(resolved_proj_dir, genre_subdir)

        else:
            raise ValueError(
                "Either proj_dir, corpus_path, or db_path_stub must be provided.\n"
                "For Google Books: db_path_stub='/path/to/Google_Books/', ngram_size=5, "
                "repo_release_id='20200217', repo_corpus_id='eng'\n"
                "For Davies corpora: corpus_path='/path/to/COHA', genre_focus=['fic']"
            )

    # Validate required parameters
    if dir_suffix is None:
        raise ValueError("dir_suffix parameter is required")
    if anchor_year is None:
        raise ValueError("anchor_year parameter is required")

    start_time = datetime.now()

    # Construct model directory. ngram_size is only used for path construction
    # in explicit/Davies modes; in stub mode the path is fully resolved above.
    if ngram_size is not None and db_path_stub is None:
        model_dir = os.path.join(resolved_proj_dir, f'{ngram_size}gram_files/models_{dir_suffix}')
    else:
        model_dir = os.path.join(resolved_proj_dir, f'models_{dir_suffix}')

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_paths = get_model_paths(model_dir)
    if not model_paths:
        raise FileNotFoundError(f"No .kv models found in {model_dir}")

    output_dir = str(Path(model_dir) / "norm_and_align")

    # Locate anchor path
    anchor_model_path = next((p for y, p in model_paths if y == anchor_year), None)
    if anchor_model_path is None:
        raise ValueError(f"Anchor model for year {anchor_year} not found.")

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

    # Compute stability weights if requested. Workers receive weights as a
    # plain dict (pickling-safe) rather than a model object.
    stability_weights = None
    if weighted_alignment:
        from .stability_weighting import load_models_for_stability_weighting, compute_stability_weights

        print("")
        print("Stability Weight Computation")
        print("═" * LINE_WIDTH)
        models_for_weighting, shared_vocab = load_models_for_stability_weighting(
            model_paths, verbose=True
        )
        stability_weights = compute_stability_weights(
            models=models_for_weighting,
            shared_vocab=shared_vocab,
            method=stability_method,
            include_frequency=include_frequency,
            frequency_weight=frequency_weight,
            verbose=True
        )
        print("")

    # All models — including the anchor — go through process_model so that
    # normalization, vocab filtering, and save logic are unified.
    tasks = [
        (y, p, anchor_year, anchor_model_path, dir_suffix, stability_weights)
        for y, p in model_paths
    ]

    print("Processing Models")
    print("═" * LINE_WIDTH)
    with Pool(processes=workers) as pool:
        for _ in tqdm(
                pool.imap_unordered(process_model, tasks),
                total=len(tasks),
                desc="Aligning models",
                ncols=LINE_WIDTH,
                unit=" models"
        ):
            pass

    end_time = datetime.now()
    runtime = end_time - start_time

    from .display import print_alignment_completion
    print_alignment_completion(
        output_dir=output_dir,
        num_models=len(model_paths),
        runtime=runtime
    )
