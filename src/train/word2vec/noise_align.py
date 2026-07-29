"""Cross-replica alignment for one noise-ensemble (year, hyperparameter) cell.

Reuses the existing W2VModel Procrustes machinery
(`ngramprep/common/w2v_model.py`) that already aligns models across years --
here it aligns REPLICATE models of the SAME year/corpus instead. One replica
is picked as a fixed reference and every other replica is aligned to it in a
single pass (not a multi-pass generalized-Procrustes consensus -- see the
module docstring in `noise_finalize.py` for why).
"""

import numpy as np
from ngramprep.common.w2v_model import W2VModel

__all__ = ["align_ensemble_cell", "SPECIAL_TOKENS"]

# Matches the special-token exclusion list already used in
# stability_weighting.py's load_models_for_stability_weighting(), for the same
# reason: these tokens are semantically unstable and shouldn't influence
# alignment or be reported as if they had meaningful noise statistics.
SPECIAL_TOKENS = {
    '<UNK>', '<unk>', '<PAD>', '<pad>',
    '<S>', '</S>', '<BOS>', '<EOS>',
    '<MASK>', '<mask>',
}


def align_ensemble_cell(ordered_paths, exclude_special_tokens=True):
    """
    Align every replica in one noise-ensemble cell to a single fixed reference
    replica (the first in `ordered_paths`), restricted to the vocabulary
    shared by ALL replicas.

    Args:
        ordered_paths (list): List of (key, path) tuples, where key is e.g.
                             (corpus_replicate, seed_repeat). Order determines
                             which replica is the fixed reference (the first).
        exclude_special_tokens (bool): Drop tokens like <UNK> from the anchor
                                      vocabulary before fitting/applying
                                      alignment. Default: True.

    Returns:
        tuple:
            anchor_vocab (set[str]): Vocabulary shared by every replica -- the
                only tokens this ensemble can report noise statistics for
                (see the scoping note in `noise_finalize.py`).
            models (dict[key -> W2VModel]): Each replica's W2VModel, already
                normalized and aligned, with `.filtered_vectors` populated
                over exactly `anchor_vocab`.
            raw_norms (dict[key -> dict[str, float]]): Pre-normalization
                vector norms per token per replica (for the norm_sd
                diagnostic computed downstream).

    Raises:
        ValueError: If no vocabulary is shared across all replicas.
    """
    models = {}
    raw_norms = {}
    vocabs = []

    for key, path in ordered_paths:
        m = W2VModel(path)
        norms = np.linalg.norm(m.model.vectors, axis=1)
        raw_norms[key] = dict(zip(m.model.index_to_key, (float(n) for n in norms)))
        m.normalize()
        models[key] = m
        vocabs.append(m.extract_vocab())

    anchor_vocab = set.intersection(*vocabs) if vocabs else set()
    if exclude_special_tokens:
        anchor_vocab -= SPECIAL_TOKENS

    if not anchor_vocab:
        raise ValueError(
            "No vocabulary is shared across all replicas in this noise-ensemble "
            "cell; cannot align or compute noise statistics."
        )

    for m in models.values():
        m.filter_vocab(anchor_vocab)

    ref_key = ordered_paths[0][0]
    reference = models[ref_key]

    for key, m in models.items():
        if key == ref_key:
            continue
        m.align_to(reference, weights=None)

    return anchor_vocab, models, raw_norms
