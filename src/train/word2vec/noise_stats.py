"""Per-token noise-statistic computation from an aligned noise-ensemble cell.

Given aligned, unit-normalized replica vectors (from `noise_align.py`) for
every token in the shared anchor vocabulary, computes the ensemble-mean
direction and Euclidean dispersion statistics that become the delivered
model's per-token `noise` / `noise_seed` / `noise_corpus` expandos.

All dispersion measures operate on unit-normalized (directional) vectors --
this is the package's default error space (see design notes: directional
error is what matters for cosine similarity, nearest neighbors, and
contrast-projection style analyses, and avoids conflating semantic
instability with raw vector-norm instability, which is tracked separately as
`norm_sd`).
"""

import numpy as np

__all__ = ["compute_token_noise_stats"]


def compute_token_noise_stats(anchor_vocab, models, raw_norms, n_corpus_replicates, n_seed_repeats):
    """
    Compute per-token noise statistics from an aligned replica ensemble.

    Args:
        anchor_vocab (set[str]): Tokens to compute statistics for.
        models (dict[(b, s) -> W2VModel]): Aligned, unit-normalized replicas,
                                          each with `.filtered_vectors`
                                          populated over `anchor_vocab`.
        raw_norms (dict[(b, s) -> dict[str, float]]): Pre-normalization vector
                                                      norms per token per
                                                      replica.
        n_corpus_replicates (int): B, number of corpus replicates.
        n_seed_repeats (int): S, number of training-seed repeats per replicate.

    Returns:
        tuple:
            stats (dict[str -> dict]): Per-token `noise`, `noise_seed`,
                `noise_corpus`, `norm_sd`, `n_reps`, `presence`.
            mean_vectors (dict[str -> np.ndarray]): Aligned ensemble-mean
                (unit-normalized-input) vector per token -- becomes the
                delivered model's `.vectors` row for that token.
    """
    B, S = n_corpus_replicates, n_seed_repeats
    stats = {}
    mean_vectors = {}

    for token in anchor_vocab:
        # arr[b, s] = aligned unit vector for this token from replica (b, s)
        arr = np.stack([
            np.stack([models[(b, s)].filtered_vectors[token] for s in range(S)])
            for b in range(B)
        ])  # shape (B, S, dim)

        u_bar = arr.mean(axis=(0, 1))
        n_total = B * S

        if n_total > 1:
            total_sq = float(np.sum(np.linalg.norm(arr - u_bar, axis=2) ** 2))
            noise_total = float(np.sqrt(total_sq / (n_total - 1)))
        else:
            noise_total = float('nan')

        # Per-corpus-replicate mean across seed repeats; used by both the
        # noise_seed and noise_corpus computations below regardless of S.
        u_bar_b = arr.mean(axis=1)  # shape (B, dim)

        noise_seed = float('nan')
        if S > 1:
            seed_sq = float(np.sum(np.linalg.norm(arr - u_bar_b[:, None, :], axis=2) ** 2))
            noise_seed = float(np.sqrt(seed_sq / (B * (S - 1))))

        noise_corpus = float('nan')
        if B > 1:
            corpus_sq = float(np.sum(np.linalg.norm(u_bar_b - u_bar, axis=1) ** 2))
            raw_corpus_var = corpus_sq / (B - 1)
            seed_var = (noise_seed ** 2) if S > 1 else 0.0
            adjusted_var = max(0.0, raw_corpus_var - (seed_var / S if S > 0 else 0.0))
            noise_corpus = float(np.sqrt(adjusted_var))

        norms = np.array([
            raw_norms[(b, s)].get(token, np.nan)
            for b in range(B) for s in range(S)
        ])
        valid_norms = norms[~np.isnan(norms)]
        norm_sd = float(np.std(valid_norms, ddof=1)) if len(valid_norms) > 1 else float('nan')

        stats[token] = dict(
            noise=noise_total,
            noise_seed=noise_seed,
            noise_corpus=noise_corpus,
            norm_sd=norm_sd,
            n_reps=n_total,
            presence=1.0,
        )
        mean_vectors[token] = u_bar

    return stats, mean_vectors
