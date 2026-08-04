"""Noise-ensemble training orchestration for Word2Vec models.

Trains a crossed (corpus_replicate x seed_repeat) grid of Word2Vec fits for
every (year, hyperparameter cell) in the ordinary grid, using deterministically
seeded corpus resampling (see `noise.py`) and Word2Vec training seeds so every
replica is reproducible from (noise_seed, year, corpus_replicate, seed_repeat)
alone.

Scope: this module only produces the raw per-replica `.kv` files (one per
corpus_replicate x seed_repeat, per grid cell) under a dedicated `_noise/`
subtree of the ordinary model directory. It does NOT perform cross-replica
alignment, noise-statistic computation, or final single-.kv assembly with
`noise`/`noise_seed`/`noise_corpus` expandos -- those are a separate,
not-yet-implemented phase. Ordinary (non-noise) training in `train.py` is
completely unaffected by this module.

Noise mode always requires use_corpus_file=True (validated in `noise.py`):
corpus resampling is only frozen/reproducible once materialized to a static
text file, since Word2Vec's corpus_file path reads that file exactly once
per replica regardless of `epochs`.
"""

import fcntl
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

from gensim.models import KeyedVectors
from tqdm import tqdm

try:
    from setproctitle import setproctitle
except ImportError:
    setproctitle = None

from .config import ensure_iterable
from .model import create_corpus_file, train_word2vec
from .worker import configure_logging

__all__ = ["train_noise_ensemble"]

_CORPUS_FILE_PATTERN = re.compile(
    r'^w2v_noise_corpus_y(\d+)_wb(\w+)_uk(\w+)_c(\d+)_.+\.txt$'
)


def _cell_name(year, weight_by, vector_size, window, min_count, sg, epochs):
    return (
        f"y{year}_wb{weight_by}_vs{vector_size:03d}_w{window:03d}_"
        f"mc{min_count:03d}_sg{sg}_e{epochs:03d}"
    )


def _replica_is_valid(path):
    """Mirrors `train._is_model_valid`: cheap existence/size check, then a
    real load attempt so partial/corrupted replica files get retrained."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            return False
        KeyedVectors.load(path)
        return True
    except Exception:
        return False


def _update_ensemble_index(noise_root, resampling, n_corpus_replicates, n_seed_repeats,
                            noise_seed, error_space, new_cells):
    """Merge `new_cells` into `ensemble_index.json` instead of overwriting it,
    so concurrent per-year calls (e.g. one Slurm array task per year, all
    sharing the same noise_root) accumulate cells rather than each clobbering
    the others' entries. Lock-protected since writers run as separate
    concurrent processes; write is atomic (temp file + os.replace)."""
    index_path = os.path.join(noise_root, "ensemble_index.json")
    with open(index_path + ".lock", "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            cells = set()
            if os.path.exists(index_path):
                try:
                    with open(index_path) as f:
                        cells.update(json.load(f).get("cells", []))
                except (json.JSONDecodeError, OSError):
                    pass
            cells.update(new_cells)

            tmp_path = index_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(dict(
                    resampling=resampling,
                    n_corpus_replicates=n_corpus_replicates,
                    n_seed_repeats=n_seed_repeats,
                    noise_seed=noise_seed,
                    error_space=error_space,
                    cells=sorted(cells),
                ), f, indent=2)
            os.replace(tmp_path, index_path)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _train_one_replica(
    db_path, corpus_file_path, year, weight_by, vector_size, window,
    min_count, approach, epochs, workers, unk_mode, training_seed,
    replica_path, log_dir, cell_name, corpus_replicate, seed_repeat,
):
    """Train and save one (corpus_replicate, seed_repeat) replica. Runs in a
    worker process, submitted with keyword arguments matching this signature."""
    sg = 1 if approach == 'skip-gram' else 0
    tag = f"{cell_name}_c{corpus_replicate:03d}_s{seed_repeat:03d}"

    if setproctitle is not None:
        try:
            setproctitle(f"ngt:noise_{tag}")
        except Exception:
            pass

    logger = configure_logging(log_dir, filename=f"w2v_noise_{tag}.log")

    try:
        logger.info(
            f"Training noise replica {tag} "
            f"(training_seed={training_seed}, corpus_file={corpus_file_path})..."
        )

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
            unk_mode=unk_mode,
            use_corpus_file=True,
            corpus_file_path=corpus_file_path,
            seed=training_seed,
        )

        os.makedirs(os.path.dirname(replica_path), exist_ok=True)
        model.wv.save(replica_path)
        logger.info(f"Replica {tag} saved to {replica_path}.")
    except Exception as e:
        logger.error(f"Error training noise replica {tag}: {e}", exc_info=True)
        raise  # Re-raise so the main process knows the task failed.


def train_noise_ensemble(
    db_path, model_dir, log_dir, years, year_step, weight_by, vector_size,
    window, min_count, approach, epochs, noise_config, max_parallel_models,
    max_corpus_workers, workers_per_model, unk_mode, temp_dir,
    reuse_corpus_files, mode='resume',
):
    """
    Train a full noise ensemble: for every year and every resolved
    hyperparameter cell, train `noise_config.n_corpus_replicates x
    noise_config.n_seed_repeats` replicas, saved under
    `model_dir/_noise/<ensemble dirname>/<cell name>/replicas/`.

    Does not touch or overwrite ordinary (non-noise) `.kv` files directly in
    `model_dir`.

    Returns:
        str: Path to the noise-ensemble root directory
             (`model_dir/_noise/<ensemble dirname>`).
    """
    weight_by = ensure_iterable(weight_by)
    vector_size = ensure_iterable(vector_size)
    window = ensure_iterable(window)
    min_count = ensure_iterable(min_count)
    approach = ensure_iterable(approach)
    epochs = ensure_iterable(epochs)

    years_range = range(years[0], years[1] + 1, year_step)
    param_combinations = list(
        product(weight_by, vector_size, window, min_count, approach, epochs)
    )

    noise_base = os.path.join(model_dir, "_noise")
    noise_root = os.path.join(noise_base, noise_config.dirname)

    if mode == 'restart':
        if os.path.exists(noise_root):
            shutil.rmtree(noise_root, ignore_errors=True)
        os.makedirs(noise_root, exist_ok=True)
    elif mode == 'new':
        if os.path.exists(noise_root):
            raise FileExistsError(
                f"Mode 'new' requires a non-existent noise-ensemble directory.\n"
                f"Found existing: {noise_root}\n"
                f"Use mode='resume' to continue or mode='restart' to erase and start over."
            )
        os.makedirs(noise_root, exist_ok=True)
    elif mode == 'resume':
        os.makedirs(noise_root, exist_ok=True)
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Must be 'resume', 'restart', or 'new'.")

    B = noise_config.n_corpus_replicates
    S = noise_config.n_seed_repeats

    # Ensure temp_dir exists before any corpus file gets written into it --
    # tempfile.mkstemp(dir=temp_dir) raises FileNotFoundError otherwise (the
    # non-noise train_models() path does this same normalize+makedirs step).
    if temp_dir:
        temp_dir = os.path.abspath(os.path.expanduser(temp_dir))
        os.makedirs(temp_dir, exist_ok=True)

    # ---- Step 1: materialize (or reuse) one corpus replicate file per
    #      (year, weight_by, unk_mode, corpus_replicate) ------------------
    corpus_needed = {
        (year, wb_val, unk_mode, b)
        for year in years_range
        for wb_val in weight_by
        for b in range(B)
    }

    corpus_file_map = {}
    if reuse_corpus_files and temp_dir and os.path.exists(temp_dir):
        for entry in os.listdir(temp_dir):
            m = _CORPUS_FILE_PATTERN.match(entry)
            if m:
                key = (int(m.group(1)), m.group(2), m.group(3), int(m.group(4)))
                if key in corpus_needed and key not in corpus_file_map:
                    corpus_file_map[key] = os.path.join(temp_dir, entry)

    still_needed = corpus_needed - corpus_file_map.keys()
    try:
        if still_needed:
            print(
                f"Creating {len(still_needed)} noise-ensemble corpus replicate "
                f"file(s)...", flush=True
            )
            n_workers = min(len(still_needed), max_corpus_workers)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {}
                for (year, wb_val, uk_val, b) in still_needed:
                    resample_seed = noise_config.resample_seed_for(year, b)
                    futures[executor.submit(
                        create_corpus_file,
                        db_path=db_path,
                        year=year,
                        weight_by=wb_val,
                        unk_mode=uk_val,
                        temp_dir=temp_dir,
                        resample_seed=resample_seed,
                        file_prefix=f"w2v_noise_corpus_y{year}_wb{wb_val}_uk{uk_val}_c{b:03d}_",
                    )] = (year, wb_val, uk_val, b)
                for future in as_completed(futures):
                    key = futures[future]
                    corpus_file_map[key] = future.result()
            print("", flush=True)

        # ---- Step 2: expand and run the (year, cell, b, s) replica grid ----
        replica_tasks = []
        cell_names = []
        for year in years_range:
            for params in param_combinations:
                wb_val, vs, win, mc, appr, ep = params
                sg = 1 if appr == 'skip-gram' else 0
                cell_name = _cell_name(year, wb_val, vs, win, mc, sg, ep)
                cell_names.append(cell_name)
                cell_dir = os.path.join(noise_root, cell_name)
                replicas_dir = os.path.join(cell_dir, "replicas")
                os.makedirs(replicas_dir, exist_ok=True)

                manifest_entries = []
                for b in range(B):
                    corpus_file_path = corpus_file_map[(year, wb_val, unk_mode, b)]
                    for s in range(S):
                        training_seed = noise_config.training_seed_for(year, b, s)
                        replica_path = os.path.join(replicas_dir, f"c{b:03d}_s{s:03d}.kv")
                        manifest_entries.append(dict(
                            corpus_replicate=b, seed_repeat=s,
                            training_seed=training_seed, path=replica_path,
                        ))
                        if _replica_is_valid(replica_path):
                            continue
                        replica_tasks.append(dict(
                            db_path=db_path, corpus_file_path=corpus_file_path,
                            year=year, weight_by=wb_val, vector_size=vs,
                            window=win, min_count=mc, approach=appr, epochs=ep,
                            workers=workers_per_model, unk_mode=unk_mode,
                            training_seed=training_seed, replica_path=replica_path,
                            log_dir=log_dir, cell_name=cell_name,
                            corpus_replicate=b, seed_repeat=s,
                        ))

                with open(os.path.join(cell_dir, "manifest.json"), "w") as f:
                    json.dump(dict(
                        year=year, weight_by=wb_val, vector_size=vs, window=win,
                        min_count=mc, approach=appr, epochs=ep,
                        n_corpus_replicates=B, n_seed_repeats=S,
                        resampling=noise_config.resampling, noise_seed=noise_config.seed,
                        error_space=noise_config.error_space,
                        replicas=manifest_entries,
                    ), f, indent=2)

        print(
            f"Training {len(replica_tasks)} noise-ensemble replica(s) "
            f"({B} corpus replicate(s) x {S} seed repeat(s) per cell)...",
            flush=True
        )

        with tqdm(total=len(replica_tasks), desc="Training noise replicas",
                  unit=" replicas") as pbar:
            with ProcessPoolExecutor(max_workers=max_parallel_models) as executor:
                futures = [
                    executor.submit(_train_one_replica, **task)
                    for task in replica_tasks
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        tqdm.write(f"\nNoise replica task failed: {e}")
                    pbar.update(1)
    finally:
        if not reuse_corpus_files:
            for path in corpus_file_map.values():
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception:
                        pass

    _update_ensemble_index(
        noise_root, noise_config.resampling, B, S, noise_config.seed,
        noise_config.error_space, cell_names,
    )

    print(
        f"Noise ensemble replicas written to: {noise_root}\n"
        f"(Alignment, per-token noise statistics, and final single-.kv "
        f"assembly into {model_dir} are a separate, not-yet-implemented step.)"
    )

    return noise_root
