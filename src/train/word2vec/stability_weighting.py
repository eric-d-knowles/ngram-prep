"""
Stability weighting for Procrustes alignment.

Computes per-word stability scores across time periods to use as weights
in weighted Procrustes alignment, so that semantically stable words
contribute more to estimating the rotation matrix.
"""
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from tqdm import tqdm

from ngramprep.common.w2v_model import W2VModel
from .display import LINE_WIDTH


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize a word -> score dict to [0, 1] via rank-based (quantile) scaling.

    Each word's score is replaced by its fractional rank across all words,
    mapping the lowest score to 0.0 and the highest to 1.0. This is robust
    to outliers in a way that min-max scaling is not — a single extremely
    stable or unstable word cannot compress the rest of the distribution.
    """
    if not scores:
        return scores

    sorted_words = sorted(scores, key=lambda w: scores[w])
    n = len(sorted_words)

    if n == 1:
        return {sorted_words[0]: 1.0}

    return {word: rank / (n - 1) for rank, word in enumerate(sorted_words)}


def compute_local_stability(
        models: List[Tuple[int, W2VModel]],
        shared_vocab: Set[str]
) -> Dict[str, float]:
    """
    Compute local stability (average year-over-year cosine similarity) for each word.

    Words with high local stability maintain consistent meanings across consecutive
    time periods.

    Args:
        models: List of (year, W2VModel) tuples, sorted by year.
        shared_vocab: Set of words present in all models.

    Returns:
        Dict mapping word -> average cosine similarity across consecutive years
        (higher = more stable).
    """
    word_similarities = defaultdict(list)

    for i in range(len(models) - 1):
        year1, model1 = models[i]
        year2, model2 = models[i + 1]

        for word in shared_vocab:
            vec1 = model1.model[word]
            vec2 = model2.model[word]
            # Cosine similarity (assumes normalized vectors)
            word_similarities[word].append(np.dot(vec1, vec2))

    return {word: np.mean(sims) for word, sims in word_similarities.items()}


def compute_global_stability(
        models: List[Tuple[int, W2VModel]],
        shared_vocab: Set[str]
) -> Dict[str, float]:
    """
    Compute global stability (inverse of variance from mean embedding) for each word.

    Words with low variance across time periods are globally stable.

    Args:
        models: List of (year, W2VModel) tuples.
        shared_vocab: Set of words present in all models.

    Returns:
        Dict mapping word -> stability score (higher = more stable),
        computed as 1 / (1 + variance_of_distances_from_mean).
    """
    word_embeddings = defaultdict(list)

    for year, model in models:
        for word in shared_vocab:
            word_embeddings[word].append(model.model[word])

    word_stability = {}
    for word, embeddings in word_embeddings.items():
        embeddings = np.array(embeddings)
        mean_embedding = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - mean_embedding, axis=1)
        variance = np.var(distances)
        # Add 1 to denominator to avoid division by zero
        word_stability[word] = 1.0 / (1.0 + variance)

    return word_stability


def compute_frequency_stability(
        models: List[Tuple[int, W2VModel]],
        shared_vocab: Set[str]
) -> Dict[str, float]:
    """
    Compute frequency stability (inverse of coefficient of variation) for each word.

    NOTE: Requires models to have word count information. Returns uniform scores
    if counts are unavailable.

    Args:
        models: List of (year, W2VModel) tuples.
        shared_vocab: Set of words present in all models.

    Returns:
        Dict mapping word -> stability score (higher = more stable).
    """
    word_counts = defaultdict(list)

    has_counts = False
    for year, model in models:
        if hasattr(model.model, 'get_vecattr'):
            has_counts = True
            for word in shared_vocab:
                try:
                    count = model.model.get_vecattr(word, 'count')
                    word_counts[word].append(count)
                except (KeyError, AttributeError):
                    pass

    if not has_counts:
        return {word: 1.0 for word in shared_vocab}

    word_stability = {}
    for word, counts in word_counts.items():
        if not counts or len(counts) < 2:
            word_stability[word] = 0.0
            continue

        counts = np.array(counts)
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        if mean_count == 0:
            word_stability[word] = 0.0
        else:
            cv = std_count / mean_count
            # Lower CV = higher stability
            word_stability[word] = 1.0 / (1.0 + cv)

    return word_stability


def compute_mean_frequency(
        models: List[Tuple[int, W2VModel]],
        shared_vocab: Set[str]
) -> Dict[str, float]:
    """
    Compute mean frequency (log-scaled) for each word across all models.

    More frequent words have more reliable embeddings and should receive
    higher weights.

    Args:
        models: List of (year, W2VModel) tuples.
        shared_vocab: Set of words present in all models.

    Returns:
        Dict mapping word -> log-scaled frequency score (higher = more frequent).
    """
    word_counts = defaultdict(list)

    has_counts = False
    for year, model in models:
        if hasattr(model.model, 'get_vecattr'):
            has_counts = True
            for word in shared_vocab:
                try:
                    count = model.model.get_vecattr(word, 'count')
                    word_counts[word].append(count)
                except (KeyError, AttributeError):
                    pass

    if not has_counts:
        return {word: 1.0 for word in shared_vocab}

    word_frequency = {}
    for word, counts in word_counts.items():
        if not counts:
            word_frequency[word] = 0.0
            continue
        mean_count = np.mean(counts)
        # log(1 + count) for numerical stability; suppresses dominance of
        # very frequent words
        word_frequency[word] = np.log1p(mean_count) if mean_count > 0 else 0.0

    return word_frequency


def compute_stability_weights(
        models: List[Tuple[int, W2VModel]],
        shared_vocab: Set[str],
        method: str = 'local_stability',
        include_frequency: bool = True,
        frequency_weight: float = 0.3,
        verbose: bool = True
) -> Dict[str, float]:
    """
    Compute stability weights for all words in shared vocabulary.

    Returns a dictionary mapping each word to its stability score for use
    as weights in weighted Procrustes alignment. More stable words receive
    higher weights and contribute more to the rotation estimate.

    Args:
        models: List of (year, W2VModel) tuples, sorted by year.
        shared_vocab: Set of words present in all models.
        method: Stability metric to use:
            - 'local_stability': Year-over-year cosine similarity
            - 'global_stability': Variance from mean embedding
            - 'frequency_stability': Coefficient of variation in frequency
            - 'combined': Equal-weighted combination of all three metrics
        include_frequency: If True, incorporate mean frequency into weights.
                          More frequent words have more reliable embeddings.
        frequency_weight: Weight for frequency component (0.0-1.0).
                         Final weight = (1-α)*stability + α*frequency.
                         Default 0.3 gives 70% stability, 30% frequency.
        verbose: If True, print progress information.

    Returns:
        Dict mapping all shared vocab words to their stability scores.
    """
    if verbose:
        method_desc = method
        if include_frequency:
            method_desc += f" + frequency (weight={frequency_weight:.2f})"
        print(f"  Method:  {method_desc}")
        print(f"  Vocab:   {len(shared_vocab):,} words   Models: {len(models)}")

    # --- Compute stability scores ---
    if method == 'local_stability':
        stability_scores = compute_local_stability(models, shared_vocab)

    elif method == 'global_stability':
        stability_scores = compute_global_stability(models, shared_vocab)

    elif method == 'frequency_stability':
        stability_scores = compute_frequency_stability(models, shared_vocab)

    elif method == 'combined':
        if verbose:
            print("  Computing combined stability metric...")
        local = _normalize_scores(compute_local_stability(models, shared_vocab))
        global_s = _normalize_scores(compute_global_stability(models, shared_vocab))
        freq_stab = _normalize_scores(compute_frequency_stability(models, shared_vocab))

        # Average the three normalized metrics; result is already in [0, 1]
        # so the outer _normalize_scores call below is a no-op but kept for
        # uniformity across all method branches.
        stability_scores = {
            word: (local[word] + global_s[word] + freq_stab[word]) / 3.0
            for word in shared_vocab
        }

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Choose from: local_stability, global_stability, frequency_stability, combined"
        )

    stability_scores = _normalize_scores(stability_scores)

    # --- Incorporate frequency if requested ---
    if include_frequency:
        if verbose:
            print("  Computing mean frequency scores...")
        frequency_scores = _normalize_scores(compute_mean_frequency(models, shared_vocab))

        # Final weight = (1-α)*stability + α*frequency
        final_scores = {
            word: (1 - frequency_weight) * stability_scores[word] +
                  frequency_weight * frequency_scores[word]
            for word in shared_vocab
        }
    else:
        final_scores = stability_scores

    if verbose:
        sorted_words = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"  Weight range: {sorted_words[-1][1]:.4f} (lowest) to "
              f"{sorted_words[0][1]:.4f} (highest)")
        print(f"\n  Top 10 highest-weighted words:")
        for word, score in sorted_words[:10]:
            print(f"    {word:20s} {score:.4f}")

    return final_scores


def load_models_for_stability_weighting(
        model_paths: List[Tuple[int, str]],
        verbose: bool = True,
        exclude_special_tokens: bool = True
) -> Tuple[List[Tuple[int, W2VModel]], Set[str]]:
    """
    Load all models and compute their shared vocabulary for stability weight
    computation.

    Args:
        model_paths: List of (year, path) tuples.
        verbose: If True, show progress bar.
        exclude_special_tokens: If True, exclude special tokens like <UNK> from
                               shared vocab. Special tokens are semantically
                               unstable and should not influence alignment.

    Returns:
        Tuple of (loaded_models, shared_vocab).
    """
    models = []

    if verbose:
        print(f"  Loading {len(model_paths)} models...")
        iterator = tqdm(
            model_paths,
            desc="  Loading models",
            ncols=LINE_WIDTH,
            unit=" models"
        )
    else:
        iterator = model_paths

    for year, path in iterator:
        model = W2VModel(path)
        # W2VModel.normalize() handles the read-only copy internally
        model = model.normalize()
        models.append((year, model))

    if verbose:
        print("  Computing shared vocabulary...")

    vocabs = [set(model.vocab) for year, model in models]
    shared_vocab = set.intersection(*vocabs)

    if exclude_special_tokens:
        special_tokens = {
            '<UNK>', '<unk>', '<PAD>', '<pad>',
            '<S>', '</S>', '<BOS>', '<EOS>',
            '<MASK>', '<mask>'
        }
        original_size = len(shared_vocab)
        shared_vocab = {word for word in shared_vocab if word not in special_tokens}

        if verbose and original_size > len(shared_vocab):
            excluded = original_size - len(shared_vocab)
            print(f"  Excluded {excluded} special token(s) from shared vocabulary")

    if verbose:
        print(f"  Shared vocabulary: {len(shared_vocab):,} words")

    return models, shared_vocab
