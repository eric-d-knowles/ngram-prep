"""Finalize a completed noise ensemble into ordinary, single-.kv deliverables.

Public entry point: `finalize_noise_ensemble()`. Call this AFTER a noise
ensemble's replicas have finished training (e.g. after a SLURM array job
submitted via `train_noise_ensemble` completes) -- mirrors how
`evaluate_word2vec_models()` / `normalize_and_align_vectors()` are separate,
explicitly-invoked steps that run after `build_word2vec_models()` finishes
training.

For each (year, hyperparameter cell), this:
  1. Aligns every corpus_replicate x seed_repeat replica to one fixed
     reference replica (single-pass Procrustes, not a multi-pass generalized
     consensus -- a scoping simplification, see below).
  2. Computes per-token noise/noise_seed/noise_corpus/norm_sd statistics over
     the vocabulary shared by every replica in the cell.
  3. Assembles an ordinary single-file KeyedVectors (ensemble-mean vectors +
     concise per-token expandos + model-level metadata) at the ORDINARY
     model path `model_dir/w2v_{cell_name}.kv` -- the same path a non-noise
     training run would use.

Known scope limitations (v1):
  - The alignment reference is a single fixed replica (the first,
    deterministically ordered by (corpus_replicate, seed_repeat)), not an
    iterative generalized-Procrustes consensus across all replicas. Simpler,
    and reuses existing, tested single-pass alignment code (W2VModel.align_to,
    already used for cross-year alignment); a consensus reference would be
    more rotation-robust but is not implemented here.
  - The reported vocabulary is the STRICT intersection across all replicas in
    the cell (the "anchor vocab"). Tokens present in only some replicas (e.g.
    dropped by resampling + min_count in a few replicas) get no noise
    statistics and are absent from the delivered model entirely -- there is
    no partial-presence support in this version. `presence` is therefore
    always 1.0 and `n_reps` always `n_corpus_replicates * n_seed_repeats` for
    every token that appears in the output at all.
"""

import json
import os
from itertools import product

import numpy as np
from gensim.models import KeyedVectors

from .config import ensure_iterable
from .noise_align import align_ensemble_cell
from .noise_stats import compute_token_noise_stats
from .noise_train import _cell_name, _replica_is_valid

__all__ = ["finalize_noise_ensemble", "finalize_noise_cell", "assemble_noise_kv"]


def assemble_noise_kv(anchor_vocab, mean_vectors, stats, vector_size, metadata, out_path):
    """
    Build and save the final single-.kv noise-ensemble deliverable.

    Args:
        anchor_vocab (set[str]): Tokens to include in the output.
        mean_vectors (dict[str -> np.ndarray]): Ensemble-mean vector per token.
        stats (dict[str -> dict]): Per-token noise statistics (see
                                  `compute_token_noise_stats`).
        vector_size (int): Embedding dimensionality.
        metadata (dict): Model-level metadata, attached as
                        `kv.lexichron_metadata`.
        out_path (str): Destination `.kv` path.

    Returns:
        str: `out_path`.
    """
    tokens = sorted(anchor_vocab)
    vectors = np.vstack([mean_vectors[t] for t in tokens]).astype(np.float32)

    kv = KeyedVectors(vector_size)
    kv.add_vectors(tokens, vectors)

    for t in tokens:
        s = stats[t]
        kv.set_vecattr(t, 'noise', s['noise'])
        kv.set_vecattr(t, 'noise_seed', s['noise_seed'])
        kv.set_vecattr(t, 'noise_corpus', s['noise_corpus'])
        kv.set_vecattr(t, 'norm_sd', s['norm_sd'])
        kv.set_vecattr(t, 'n_reps', s['n_reps'])
        kv.set_vecattr(t, 'presence', s['presence'])

    kv.lexichron_metadata = metadata

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # separately=[] forces everything (including the .vectors array) into ONE
    # file regardless of size. gensim's default sep_limit (10MB) would
    # otherwise externalize .vectors to a sidecar file for any vocabulary
    # bigger than roughly 8-9k tokens at vector_size=300 -- verified via a
    # round-trip save/load test with a deliberately oversized array.
    kv.save(out_path, separately=[])
    return out_path


def finalize_noise_cell(cell_dir, model_dir, exclude_special_tokens=True):
    """
    Align, compute noise statistics for, and assemble the final .kv for one
    completed noise-ensemble cell.

    Args:
        cell_dir (str): e.g. `model_dir/_noise/<ensemble dirname>/<cell_name>`.
        model_dir (str): Ordinary model directory; the output is written to
                        `model_dir/w2v_{cell_name}.kv`.
        exclude_special_tokens (bool): Passed to `align_ensemble_cell`.

    Returns:
        str: Path to the finalized `.kv` file.

    Raises:
        FileNotFoundError: If the cell's manifest.json is missing.
        RuntimeError: If any replica listed in the manifest is missing or
            fails to load (the ensemble is incomplete; finish training first).
    """
    manifest_path = os.path.join(cell_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No manifest.json found in {cell_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    B = manifest["n_corpus_replicates"]
    S = manifest["n_seed_repeats"]
    cell_name = os.path.basename(cell_dir.rstrip("/"))

    replica_paths = {}
    for entry in manifest["replicas"]:
        key = (entry["corpus_replicate"], entry["seed_repeat"])
        path = entry["path"]
        if not _replica_is_valid(path):
            raise RuntimeError(
                f"Replica missing or invalid: {path}. This noise ensemble is "
                f"incomplete for cell '{cell_name}' -- finish training "
                f"(train_noise_ensemble) before finalizing."
            )
        replica_paths[key] = path

    ordered_paths = [
        ((b, s), replica_paths[(b, s)])
        for b in range(B) for s in range(S)
    ]

    anchor_vocab, models, raw_norms = align_ensemble_cell(
        ordered_paths, exclude_special_tokens=exclude_special_tokens
    )
    stats, mean_vectors = compute_token_noise_stats(anchor_vocab, models, raw_norms, B, S)

    metadata = {
        "schema": "lexichron.word2vec.noise.v1",
        "cell_name": cell_name,
        "year": manifest["year"],
        "weight_by": manifest["weight_by"],
        "vector_size": manifest["vector_size"],
        "window": manifest["window"],
        "min_count": manifest["min_count"],
        "approach": manifest["approach"],
        "epochs": manifest["epochs"],
        "n_corpus_replicates": B,
        "n_seed_repeats": S,
        "resampling": manifest["resampling"],
        "noise_seed": manifest["noise_seed"],
        "error_space": manifest.get("error_space", "unit"),
        "primary_noise_expando": "noise",
        "primary_noise_definition": (
            "sample SD of Euclidean displacement among globally aligned, "
            "unit-normalized replicate vectors, relative to the ensemble mean"
        ),
        "alignment_method": "single_reference_procrustes",
        "anchor_vocab_size": len(anchor_vocab),
    }

    out_path = os.path.join(model_dir, f"w2v_{cell_name}.kv")
    assemble_noise_kv(anchor_vocab, mean_vectors, stats, manifest["vector_size"], metadata, out_path)
    return out_path


def finalize_noise_ensemble(
    model_dir, years, year_step, weight_by, vector_size, window, min_count,
    approach, epochs, noise_config, exclude_special_tokens=True,
    skip_existing=True,
):
    """
    Finalize every (year, hyperparameter cell) in a completed noise ensemble
    into ordinary single-.kv deliverables at `model_dir/w2v_{cell_name}.kv`.

    Call this after `train_noise_ensemble()` (invoked via
    `build_word2vec_models(noise_enabled=True, ...)`) has finished training
    all replicas -- e.g. after a SLURM array job completes, mirroring how
    `evaluate_word2vec_models()` is a separate step run after training.

    Args:
        model_dir (str): Ordinary model directory (as returned by
                        `build_word2vec_models`).
        years, year_step, weight_by, vector_size, window, min_count, approach,
            epochs: Same grid-defining arguments passed to
            `build_word2vec_models`.
        noise_config (NoiseConfig): The resolved config (see `noise.py`); must
                                   match what was used for training.
        exclude_special_tokens (bool): Passed to `align_ensemble_cell`.
        skip_existing (bool): If True, cells whose final .kv already exists
                             and loads validly are left untouched rather than
                             recomputed. Default: True.

    Returns:
        dict[str, str or None]: Maps cell_name to its output path, or None if
            that cell failed to finalize (see printed diagnostics for why).
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

    noise_root = os.path.join(model_dir, "_noise", noise_config.dirname)
    if not os.path.exists(noise_root):
        raise FileNotFoundError(
            f"Noise-ensemble root not found: {noise_root}. Train it first via "
            f"build_word2vec_models(noise_enabled=True, ...) with a matching "
            f"noise configuration."
        )

    results = {}
    for year in years_range:
        for params in param_combinations:
            wb_val, vs, win, mc, appr, ep = params
            sg = 1 if appr == 'skip-gram' else 0
            cell_name = _cell_name(year, wb_val, vs, win, mc, sg, ep)
            cell_dir = os.path.join(noise_root, cell_name)
            out_path = os.path.join(model_dir, f"w2v_{cell_name}.kv")

            if skip_existing and _replica_is_valid(out_path):
                print(f"{cell_name}: final model already exists, skipping")
                results[cell_name] = out_path
                continue

            try:
                out_path = finalize_noise_cell(
                    cell_dir, model_dir, exclude_special_tokens=exclude_special_tokens
                )
                print(f"{cell_name}: finalized -> {out_path}")
                results[cell_name] = out_path
            except Exception as e:
                print(f"{cell_name}: FAILED to finalize: {e}")
                results[cell_name] = None

    return results
