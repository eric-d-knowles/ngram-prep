"""Noise-ensemble configuration, seeding, and validation for Word2Vec training.

This module implements the corpus-replicate x seed-repeat ensemble design used
to proxy embedding measurement error: for a given (year, hyperparameter cell),
train `n_corpus_replicates` Poisson-resampled corpus variants, each refit under
`n_seed_repeats` distinct Word2Vec training seeds. The resulting replica set
lets downstream code (not yet implemented here) separate corpus-resampling
noise from training-stochasticity noise.

This module is purely additive: nothing here is imported or exercised unless
a caller explicitly opts in via `build_word2vec_models(noise_enabled=True, ...)`.
Ordinary (non-noise) training is completely unaffected.
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "NOISE_RESAMPLING_METHODS",
    "NoiseConfig",
    "validate_noise_config",
    "derive_resample_seed",
    "derive_training_seed",
    "noise_root_dirname",
]

# Salts distinguish the two independent seed streams derived from the same
# (noise_seed, year, corpus_replicate[, seed_repeat]) coordinates. Fixed
# arbitrary constants -- never change these once ensembles have been produced
# with them, or seeds will silently stop reproducing past results.
_SALT_RESAMPLE = 0xA17E5A11
_SALT_TRAIN = 0x5EEDF00D

NOISE_RESAMPLING_METHODS = ("seed_only", "poisson_counts", "document_bootstrap")


def derive_resample_seed(noise_seed, year, corpus_replicate):
    """
    Deterministically derive the corpus-resampling seed for one
    (year, corpus_replicate) pair. This seed is reused across every
    seed_repeat for that replicate, so the corpus stays frozen while only
    the Word2Vec training seed varies.

    Args:
        noise_seed (int): Top-level ensemble seed.
        year (int): Training year.
        corpus_replicate (int): Corpus-replicate index (0-based).

    Returns:
        int: Seed for `np.random.default_rng()`.
    """
    ss = np.random.SeedSequence(
        [int(noise_seed), int(year), int(corpus_replicate), _SALT_RESAMPLE]
    )
    return int(ss.generate_state(1)[0])


def derive_training_seed(noise_seed, year, corpus_replicate, seed_repeat):
    """
    Deterministically derive the Word2Vec training seed for one
    (year, corpus_replicate, seed_repeat) triple.

    Args:
        noise_seed (int): Top-level ensemble seed.
        year (int): Training year.
        corpus_replicate (int): Corpus-replicate index (0-based).
        seed_repeat (int): Seed-repeat index within that replicate (0-based).

    Returns:
        int: Seed for gensim's `Word2Vec(seed=...)`.
    """
    ss = np.random.SeedSequence(
        [int(noise_seed), int(year), int(corpus_replicate), int(seed_repeat), _SALT_TRAIN]
    )
    return int(ss.generate_state(1)[0])


def noise_root_dirname(resampling, n_corpus_replicates, n_seed_repeats, seed):
    """Build the deterministic ensemble directory name for one noise configuration."""
    return f"{resampling}-c{n_corpus_replicates:03d}-s{n_seed_repeats:03d}-seed{seed}"


@dataclass(frozen=True)
class NoiseConfig:
    """Resolved, validated configuration for a noise-enabled training run."""

    resampling: str
    n_corpus_replicates: int
    n_seed_repeats: int
    seed: int
    error_space: str = "unit"

    @property
    def dirname(self):
        return noise_root_dirname(
            self.resampling, self.n_corpus_replicates, self.n_seed_repeats, self.seed
        )

    def resample_seed_for(self, year, corpus_replicate):
        """Corpus-resampling seed for one (year, corpus_replicate), or None if
        this configuration performs no corpus perturbation (`seed_only`)."""
        if self.resampling == "seed_only":
            return None
        return derive_resample_seed(self.seed, year, corpus_replicate)

    def training_seed_for(self, year, corpus_replicate, seed_repeat):
        """Word2Vec training seed for one (year, corpus_replicate, seed_repeat)."""
        return derive_training_seed(self.seed, year, corpus_replicate, seed_repeat)


def validate_noise_config(
    noise_enabled,
    noise_resampling,
    n_corpus_replicates,
    n_seed_repeats,
    noise_seed,
    use_corpus_file,
    noise_error_space,
):
    """
    Validate a noise-ensemble configuration.

    Args mirror the `noise_*` keyword arguments of `build_word2vec_models()`,
    plus the ordinary `use_corpus_file` flag (noise mode requires it to be
    True; see the module docstring in `noise_train.py` for why).

    Returns:
        NoiseConfig or None: A resolved config if noise_enabled=True, else None
        (after confirming no noise-only argument was set while disabled).

    Raises:
        ValueError: For any inconsistent or incomplete noise configuration.
        NotImplementedError: For `noise_resampling='document_bootstrap'`, which
            is not yet implementable from the pivoted n-gram database (it
            stores only aggregate per-ngram occurrence/document counts, not
            per-document identity).
    """
    non_default = []
    if noise_resampling != "poisson_counts":
        non_default.append("noise_resampling")
    if n_corpus_replicates != 1:
        non_default.append("n_corpus_replicates")
    if n_seed_repeats != 1:
        non_default.append("n_seed_repeats")
    if noise_seed is not None:
        non_default.append("noise_seed")
    if noise_error_space != "unit":
        non_default.append("noise_error_space")

    if not noise_enabled:
        if non_default:
            raise ValueError(
                "noise_enabled=False but noise-only argument(s) were set: "
                f"{', '.join(non_default)}. Set noise_enabled=True to use them, "
                "or remove them to train an ordinary (non-noise) model."
            )
        return None

    if noise_resampling not in NOISE_RESAMPLING_METHODS:
        raise ValueError(
            f"Invalid noise_resampling: '{noise_resampling}'. "
            f"Must be one of {NOISE_RESAMPLING_METHODS}."
        )

    if noise_resampling == "document_bootstrap":
        raise NotImplementedError(
            "noise_resampling='document_bootstrap' is not yet implemented: the "
            "pivoted n-gram database stores only aggregate per-ngram "
            "occurrence/document counts, not per-document identity, so a true "
            "document-level bootstrap cannot be constructed from it. Use "
            "'poisson_counts' (corpus-count resampling) or 'seed_only' "
            "(training-seed-only diagnostic) instead."
        )

    if noise_seed is None:
        raise ValueError(
            "noise_enabled=True requires an explicit noise_seed (no default "
            "from the clock), so noise ensembles are reproducible."
        )

    if not use_corpus_file:
        raise ValueError(
            "noise_enabled=True requires use_corpus_file=True. Noise-ensemble "
            "corpus replicates are only frozen/reproducible when materialized "
            "to a static corpus file; the streaming iterator path "
            "(use_corpus_file=False) would redraw resampling noise on every "
            "internal Word2Vec pass (vocab build + each epoch)."
        )

    if n_corpus_replicates < 1 or n_seed_repeats < 1:
        raise ValueError("n_corpus_replicates and n_seed_repeats must each be >= 1.")

    if noise_resampling == "seed_only" and n_corpus_replicates != 1:
        raise ValueError(
            "noise_resampling='seed_only' performs no corpus perturbation, so "
            "every corpus replicate would be identical. Set n_corpus_replicates=1 "
            "(all variation then comes from n_seed_repeats), or choose "
            "'poisson_counts' if you want genuinely distinct corpus replicates."
        )

    if n_corpus_replicates == 1 and n_seed_repeats == 1:
        raise ValueError(
            "noise_enabled=True requires at least one of n_corpus_replicates > 1 "
            "or n_seed_repeats > 1 (with both equal to 1 there is only one "
            "replicate, and no noise can be estimated)."
        )

    if noise_error_space not in ("unit", "raw"):
        raise ValueError(
            f"Invalid noise_error_space: '{noise_error_space}'. Must be 'unit' or 'raw'."
        )

    return NoiseConfig(
        resampling=noise_resampling,
        n_corpus_replicates=n_corpus_replicates,
        n_seed_repeats=n_seed_repeats,
        seed=noise_seed,
        error_space=noise_error_space,
    )
