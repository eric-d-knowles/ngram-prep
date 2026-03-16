import os
import re
import shutil
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from ngramprep.common.w2v_model import W2VModel
from .display import LINE_WIDTH


# Module-level globals populated once per worker process by _init_worker.
# Keeps the anchor model and weights out of the task tuple, avoiding
# repeated pickling of large objects through the pool's task queue.
_anchor_model = None
_alignment_weights = None


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


def _get_swadesh_words():
    """
    Return the NLTK Swadesh word list for English as a set of lowercase
    single-token alphabetic strings.

    Requires the NLTK Swadesh corpus to be downloaded. Run
    scripts/setup_nlp_resources.sh to install all required corpora.
    """
    from nltk.corpus import swadesh
    return {
        w.lower() for w in swadesh.words('en')
        if w.isalpha() and ' ' not in w
    }


def _get_shared_vocab(model_paths):
    """
    Compute the shared vocabulary across all models by loading only
    index_to_key, not the full vector matrices. Significantly faster
    than loading complete W2VModel instances when vectors aren't needed.
    """
    from gensim.models import KeyedVectors
    vocabs = []
    for year, path in model_paths:
        kv = KeyedVectors.load(path, mmap='r')
        vocabs.append(set(kv.index_to_key))
    return set.intersection(*vocabs)


def _init_worker(anchor_model_path, weights):
    """
    Pool initializer: load, normalize, and filter the anchor model once
    per worker process, and store the alignment weights. Both are stored
    as module-level globals so process_model can access them without any
    additional I/O, computation, or pickling per task.
    """
    global _anchor_model, _alignment_weights
    anchor = W2VModel(anchor_model_path)
    anchor = anchor.normalize()
    anchor.filter_vocab(anchor.extract_vocab())
    _anchor_model = anchor
    _alignment_weights = weights


def process_model(args):
    """
    Normalize, vocab-filter, and (for non-anchor years) align a model to
    the anchor. The anchor model and weights are accessed from module-level
    globals set by _init_worker rather than passed per task, avoiding
    repeated pickling of large objects through the pool queue.
    """
    year, model_path, anchor_year, output_path = args

    model = W2VModel(model_path)
    model = model.normalize()

    if year == anchor_year:
        # Anchor: filter to its own vocab to establish the shared vocabulary
        # baseline. All other models will be filtered to this same set.
        model.filter_vocab(model.extract_vocab())
    else:
        model.filter_vocab(_anchor_model.filtered_vocab)
        model.align_to(_anchor_model, weights=_alignment_weights)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)


def _run_alignment_pass(tasks, workers, anchor_model_path, weights, desc):
    """
    Run one alignment pass and return the output model paths.

    The anchor model and weights are passed to each worker process once
    via the pool initializer rather than once per task. Uses a spawn
    context to avoid fork-inherited lock deadlocks with NumPy and gensim
    without affecting the global multiprocessing start method.

    Args:
        tasks: List of process_model arg tuples:
               (year, model_path, anchor_year, output_path).
        workers: Number of parallel workers.
        anchor_model_path: Path to the anchor model, passed to _init_worker.
        weights: Alignment weights dict, passed to _init_worker.
        desc: tqdm description string for this pass.

    Returns:
        List of (year, output_path) tuples for all models processed.
    """
    ctx = mp.get_context('spawn')
    with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(anchor_model_path, weights)
    ) as pool:
        for _ in tqdm(
                pool.imap_unordered(process_model, tasks),
                total=len(tasks),
                desc=desc,
                ncols=LINE_WIDTH,
                unit=" models"
        ):
            pass

    return [(year, output_path) for year, _, _, output_path in tasks]


def normalize_and_align_vectors(
        proj_dir=None,
        dir_suffix=None,
        anchor_year=None,
        ngram_size=None,
        workers=None,
        corpus_path=None,
        genre_focus=None,
        alignment_method='swadesh',
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
        alignment_method: Method for aligning models. One of:
            - 'swadesh' (default): Single-pass Procrustes anchored to the
              intersection of the NLTK Swadesh list with the shared vocabulary.
              Externally grounded and simple to justify methodologically.
            - 'stability_weighted': Two-pass Procrustes. First pass aligns with
              uniform weights to remove rotation confound; stability weights are
              then computed on the aligned models and used in the second pass.
            - 'unweighted': Single-pass Procrustes over the full shared
              vocabulary with uniform weights.
        stability_method: Stability metric for 'stability_weighted' alignment.
                         One of: 'local_stability', 'global_stability',
                         'frequency_stability', 'combined'.
        include_frequency: If True, incorporate word frequency into stability
                          weights. Only used if alignment_method='stability_weighted'.
        frequency_weight: Weight for frequency component (0.0-1.0). Default 0.3.
                         Only used if alignment_method='stability_weighted' and
                         include_frequency=True.
        repo_release_id: Release date in YYYYMMDD format (e.g., "20200217")
        repo_corpus_id: Corpus identifier (e.g., "eng", "eng-fiction")
        db_path_stub: Base directory for data (e.g., "/scratch/edk202/NLP_corpora/Google_Books/")

    Example:
        >>> # Swadesh-anchored alignment (default, recommended):
        >>> normalize_and_align_vectors(
        ...     ngram_size=5,
        ...     repo_release_id='20200217',
        ...     repo_corpus_id='eng-us',
        ...     db_path_stub='/scratch/edk202/NLP_corpora/Google_Books/',
        ...     dir_suffix='final',
        ...     anchor_year=1968,
        ...     workers=50
        ... )
        >>>
        >>> # Stability-weighted alignment:
        >>> normalize_and_align_vectors(
        ...     ngram_size=5,
        ...     repo_release_id='20200217',
        ...     repo_corpus_id='eng-us',
        ...     db_path_stub='/scratch/edk202/NLP_corpora/Google_Books/',
        ...     dir_suffix='final',
        ...     anchor_year=1968,
        ...     alignment_method='stability_weighted',
        ...     stability_method='local_stability',
        ...     include_frequency=True,
        ...     frequency_weight=0.3,
        ...     workers=50
        ... )
        >>>
        >>> # Unweighted alignment:
        >>> normalize_and_align_vectors(
        ...     ngram_size=5,
        ...     repo_release_id='20200217',
        ...     repo_corpus_id='eng-us',
        ...     db_path_stub='/scratch/edk202/NLP_corpora/Google_Books/',
        ...     dir_suffix='final',
        ...     anchor_year=1968,
        ...     alignment_method='unweighted',
        ...     workers=50
        ... )
        >>>
        >>> # Davies corpus mode:
        >>> normalize_and_align_vectors(
        ...     corpus_path='/scratch/edk202/NLP_corpora/COHA',
        ...     dir_suffix='final',
        ...     anchor_year=1968,
        ...     genre_focus=['fic'],
        ...     workers=50
        ... )
        >>>
        >>> # Explicit path mode (backwards compatible):
        >>> normalize_and_align_vectors(
        ...     proj_dir='/scratch/edk202/NLP_models/Google_Books/20200217/eng-us/5gram_files',
        ...     dir_suffix='final',
        ...     anchor_year=1968,
        ...     workers=50
        ... )
    """
    if alignment_method not in ('swadesh', 'stability_weighted', 'unweighted'):
        raise ValueError(
            f"Unknown alignment_method: '{alignment_method}'. "
            f"Choose from: 'swadesh', 'stability_weighted', 'unweighted'"
        )

    # Default workers: respect SLURM allocation on HPC; fall back to CPU count
    if workers is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        workers = int(slurm_cpus) if slurm_cpus is not None else os.cpu_count()

    # --- Resolve proj_dir ---
    resolved_proj_dir = proj_dir

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
                "repo_release_id='20200217', repo_corpus_id='eng-us'\n"
                "For Davies corpora: corpus_path='/path/to/COHA', genre_focus=['fic']"
            )

    if dir_suffix is None:
        raise ValueError("dir_suffix parameter is required")
    if anchor_year is None:
        raise ValueError("anchor_year parameter is required")

    start_time = datetime.now()

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

    anchor_model_path = next((p for y, p in model_paths if y == anchor_year), None)
    if anchor_model_path is None:
        raise ValueError(f"Anchor model for year {anchor_year} not found.")

    from .display import print_alignment_header
    print_alignment_header(
        start_time=start_time,
        model_dir=model_dir,
        output_dir=output_dir,
        anchor_year=anchor_year,
        num_models=len(model_paths),
        alignment_method=alignment_method,
        stability_method=stability_method if alignment_method == 'stability_weighted' else None,
        include_frequency=include_frequency if alignment_method == 'stability_weighted' else None,
        frequency_weight=frequency_weight if alignment_method == 'stability_weighted' else None,
        workers=workers
    )

    weights = None
    pass1_model_paths = None
    final_output_dir = Path(model_dir) / "norm_and_align"

    # --- Swadesh alignment ---
    if alignment_method == 'swadesh':
        print("Swadesh Anchor Computation")
        print("═" * LINE_WIDTH)

        # Compute shared vocab in vivo from the actual trained models,
        # loading only vocabularies rather than full vector matrices.
        print("  Computing shared vocabulary...")
        shared_vocab = _get_shared_vocab(model_paths)

        swadesh_words = _get_swadesh_words()
        swadesh_anchor_vocab = swadesh_words.intersection(shared_vocab)

        print(f"  Swadesh list size:    {len(swadesh_words):,} words")
        print(f"  Shared vocab size:    {len(shared_vocab):,} words")
        print(f"  Anchor set size:      {len(swadesh_anchor_vocab):,} words")
        print("")

        # Words in anchor set get weight 1.0; all others get 0.0 so they
        # are rotated but do not contribute to estimating R.
        weights = {
            word: 1.0 if word in swadesh_anchor_vocab else 0.0
            for word in shared_vocab
        }

        print("Processing Models")
        print("═" * LINE_WIDTH)

    # --- Stability-weighted alignment ---
    elif alignment_method == 'stability_weighted':
        from .stability_weighting import (
            load_models_for_stability_weighting,
            compute_stability_weights
        )

        # Pass 1: unweighted alignment to remove rotation confound before
        # computing stability scores. Temp models written inside model_dir.
        pass1_output_dir = Path(model_dir) / "pass1_tmp"

        pass1_tasks = [
            (y, p, anchor_year, str(pass1_output_dir / Path(p).name))
            for y, p in model_paths
        ]

        print("Pass 1: Unweighted Alignment")
        print("═" * LINE_WIDTH)
        pass1_model_paths = _run_alignment_pass(
            pass1_tasks, workers, anchor_model_path,
            weights=None, desc="  Aligning models"
        )
        print("")

        # Compute stability weights on pass-1 aligned models — free of
        # rotation confound since all models now share the same vector space.
        print("Stability Weight Computation")
        print("═" * LINE_WIDTH)
        models_for_weighting, shared_vocab = load_models_for_stability_weighting(
            pass1_model_paths, verbose=True
        )
        weights = compute_stability_weights(
            models=models_for_weighting,
            shared_vocab=shared_vocab,
            method=stability_method,
            include_frequency=include_frequency,
            frequency_weight=frequency_weight,
            verbose=True
        )
        print("")

        # Update anchor to pass-1 version so pass-2 aligns against the
        # already-rotated anchor, not the raw original.
        anchor_model_path = next(
            (p for y, p in pass1_model_paths if y == anchor_year), None
        )

        # Cleanup deferred to after pass-2 completes — pass-2 reads from
        # pass1_model_paths which points to these files on disk.

        print("Pass 2: Weighted Alignment")
        print("═" * LINE_WIDTH)

    # --- Unweighted alignment ---
    else:
        print("Processing Models")
        print("═" * LINE_WIDTH)

    # Final alignment pass — weights=None for unweighted, Swadesh dict for
    # swadesh, stability dict for stability_weighted. Both anchor model and
    # weights are passed to workers once via the pool initializer. Uses a
    # spawn context to avoid fork-inherited lock deadlocks without affecting
    # the global multiprocessing start method.
    source_paths = pass1_model_paths if alignment_method == 'stability_weighted' else model_paths
    tasks = [
        (y, p, anchor_year, str(final_output_dir / Path(p).name))
        for y, p in source_paths
    ]
    _run_alignment_pass(
        tasks, workers, anchor_model_path,
        weights=weights, desc="  Aligning models"
    )

    # Clean up pass-1 tmp models now that pass-2 is complete.
    if alignment_method == 'stability_weighted':
        pass1_dir = Path(model_dir) / "pass1_tmp"
        if pass1_dir.exists():
            shutil.rmtree(pass1_dir)

    end_time = datetime.now()
    runtime = end_time - start_time

    from .display import print_alignment_completion
    print_alignment_completion(
        output_dir=output_dir,
        num_models=len(model_paths),
        runtime=runtime
    )