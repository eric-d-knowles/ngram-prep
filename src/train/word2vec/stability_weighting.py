"""
Anchor word selection for Procrustes alignment.

Identifies stable words across time periods to use as anchors for alignment,
rather than using the entire shared vocabulary.
"""
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict
from tqdm import tqdm

from ngramprep.common.w2v_model import W2VModel


def compute_local_stability(models: List[Tuple[int, W2VModel]],
                            shared_vocab: Set[str]) -> Dict[str, float]:
    """
    Compute local stability (average year-over-year cosine similarity) for each word.

    Words with high local stability maintain consistent meanings across consecutive time periods.

    Args:
        models: List of (year, W2VModel) tuples, sorted by year
        shared_vocab: Set of words present in all models

    Returns:
        Dict mapping word -> average cosine similarity across consecutive years
        (higher = more stable)
    """
    word_similarities = defaultdict(list)

    # Compare consecutive model pairs
    for i in range(len(models) - 1):
        year1, model1 = models[i]
        year2, model2 = models[i + 1]

        for word in shared_vocab:
            vec1 = model1.model[word]
            vec2 = model2.model[word]

            # Cosine similarity (assumes normalized vectors)
            similarity = np.dot(vec1, vec2)
            word_similarities[word].append(similarity)

    # Average across all consecutive pairs
    return {word: np.mean(sims) for word, sims in word_similarities.items()}


def compute_global_stability(models: List[Tuple[int, W2VModel]],
                             shared_vocab: Set[str]) -> Dict[str, float]:
    """
    Compute global stability (inverse of variance from mean embedding) for each word.

    Words with low variance across time periods are globally stable.

    Args:
        models: List of (year, W2VModel) tuples
        shared_vocab: Set of words present in all models

    Returns:
        Dict mapping word -> stability score (higher = more stable)
        Computed as 1 / (1 + variance_of_distances_from_mean)
    """
    word_embeddings = defaultdict(list)

    # Collect embeddings for each word across all years
    for year, model in models:
        for word in shared_vocab:
            word_embeddings[word].append(model.model[word])

    word_stability = {}

    for word, embeddings in word_embeddings.items():
        embeddings = np.array(embeddings)

        # Compute mean embedding
        mean_embedding = np.mean(embeddings, axis=0)

        # Compute distances from mean
        distances = np.linalg.norm(embeddings - mean_embedding, axis=1)

        # Variance of distances (lower = more stable)
        variance = np.var(distances)

        # Convert to stability score (higher = more stable)
        # Add 1 to denominator to avoid division by zero
        stability = 1.0 / (1.0 + variance)
        word_stability[word] = stability

    return word_stability


def compute_frequency_stability(models: List[Tuple[int, W2VModel]],
                                shared_vocab: Set[str]) -> Dict[str, float]:
    """
    Compute frequency stability (inverse of coefficient of variation) for each word.

    NOTE: This requires models to have word count information. If not available,
    returns uniform scores.

    Args:
        models: List of (year, W2VModel) tuples
        shared_vocab: Set of words present in all models

    Returns:
        Dict mapping word -> stability score (higher = more stable)
    """
    word_counts = defaultdict(list)

    # Try to extract word counts from models
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
        # Return uniform scores if count information not available
        return {word: 1.0 for word in shared_vocab}

    word_stability = {}

    for word, counts in word_counts.items():
        if not counts or len(counts) < 2:
            word_stability[word] = 0.0
            continue

        counts = np.array(counts)

        # Coefficient of variation (CV) = std / mean
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        if mean_count == 0:
            word_stability[word] = 0.0
        else:
            cv = std_count / mean_count
            # Convert to stability (lower CV = higher stability)
            stability = 1.0 / (1.0 + cv)
            word_stability[word] = stability

    return word_stability


def compute_mean_frequency(models: List[Tuple[int, W2VModel]],
                           shared_vocab: Set[str]) -> Dict[str, float]:
    """
    Compute mean frequency (log-scaled) for each word across all models.

    More frequent words have more reliable embeddings and should receive higher weights.

    Args:
        models: List of (year, W2VModel) tuples
        shared_vocab: Set of words present in all models

    Returns:
        Dict mapping word -> log-scaled frequency score (higher = more frequent)
    """
    word_counts = defaultdict(list)

    # Try to extract word counts from models
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
        # Return uniform scores if count information not available
        return {word: 1.0 for word in shared_vocab}

    word_frequency = {}

    for word, counts in word_counts.items():
        if not counts:
            word_frequency[word] = 0.0
            continue

        # Use log-scaled mean frequency
        # Log transform helps prevent very frequent words from dominating
        mean_count = np.mean(counts)
        if mean_count > 0:
            word_frequency[word] = np.log1p(mean_count)  # log(1 + count) for numerical stability
        else:
            word_frequency[word] = 0.0

    return word_frequency


def compute_stability_weights(models: List[Tuple[int, W2VModel]],
                              shared_vocab: Set[str],
                              method: str = 'local_stability',
                              include_frequency: bool = True,
                              frequency_weight: float = 0.3,
                              verbose: bool = True) -> Dict[str, float]:
    """
    Compute stability weights for all words in shared vocabulary.

    Returns a dictionary mapping each word to its stability score, which can be used
    as weights in weighted Procrustes alignment. More stable words receive higher weights
    and contribute more to determining the alignment transformation.

    Args:
        models: List of (year, W2VModel) tuples, sorted by year
        shared_vocab: Set of words present in all models
        method: Stability metric to use:
            - 'local_stability': Year-over-year cosine similarity
            - 'global_stability': Variance from mean embedding
            - 'frequency_stability': Coefficient of variation in frequency
            - 'combined': Weighted combination of all stability metrics
        include_frequency: If True, incorporate mean frequency into weights (recommended).
                          More frequent words have more reliable embeddings.
        frequency_weight: Weight for frequency component when include_frequency=True.
                         Final weight = (1-α)*stability + α*frequency, where α=frequency_weight.
                         Default 0.3 gives 70% weight to stability, 30% to frequency.
        verbose: If True, print progress information

    Returns:
        Dict mapping all shared vocab words to their stability scores (weights)
    """
    if verbose:
        method_desc = f"{method}"
        if include_frequency:
            method_desc += f" + frequency (weight={frequency_weight:.2f})"
        print(f"Computing stability weights using {method_desc} for {len(shared_vocab)} words across {len(models)} models...")

    # Compute stability metric
    if method == 'local_stability':
        stability_scores = compute_local_stability(models, shared_vocab)
    elif method == 'global_stability':
        stability_scores = compute_global_stability(models, shared_vocab)
    elif method == 'frequency_stability':
        stability_scores = compute_frequency_stability(models, shared_vocab)
    elif method == 'combined':
        # Combine all metrics with equal weights
        if verbose:
            print("Computing combined stability metric...")
        local = compute_local_stability(models, shared_vocab)
        global_s = compute_global_stability(models, shared_vocab)
        freq_stab = compute_frequency_stability(models, shared_vocab)

        # Normalize each metric to [0, 1]
        def normalize_scores(scores):
            values = np.array(list(scores.values()))
            min_val = values.min()
            max_val = values.max()
            if max_val == min_val:
                return {k: 1.0 for k in scores}
            return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

        local_norm = normalize_scores(local)
        global_norm = normalize_scores(global_s)
        freq_stab_norm = normalize_scores(freq_stab)

        # Average the normalized scores
        stability_scores = {
            word: (local_norm[word] + global_norm[word] + freq_stab_norm[word]) / 3.0
            for word in shared_vocab
        }
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: local_stability, global_stability, frequency_stability, combined")

    # Normalize stability scores to [0, 1]
    def normalize_scores(scores):
        values = np.array(list(scores.values()))
        min_val = values.min()
        max_val = values.max()
        if max_val == min_val:
            return {k: 1.0 for k in scores}
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

    stability_scores = normalize_scores(stability_scores)

    # Incorporate frequency if requested
    if include_frequency:
        if verbose:
            print("Computing mean frequency scores...")
        frequency_scores = compute_mean_frequency(models, shared_vocab)
        frequency_scores = normalize_scores(frequency_scores)

        # Combine: final_weight = (1-α)*stability + α*frequency
        final_scores = {
            word: (1 - frequency_weight) * stability_scores[word] + frequency_weight * frequency_scores[word]
            for word in shared_vocab
        }
    else:
        final_scores = stability_scores

    if verbose:
        sorted_words = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"Final weight range: {sorted_words[-1][1]:.4f} (lowest) to {sorted_words[0][1]:.4f} (highest)")
        print(f"\nTop 10 highest-weighted words:")
        for word, score in sorted_words[:10]:
            print(f"  {word:20s} {score:.4f}")

    return final_scores


def load_models_for_stability_weighting(model_paths: List[Tuple[int, str]],
                                        verbose: bool = True,
                                        exclude_special_tokens: bool = True) -> Tuple[List[Tuple[int, W2VModel]], Set[str]]:
    """
    Load all models and compute their shared vocabulary for stability weight computation.

    Args:
        model_paths: List of (year, path) tuples
        verbose: If True, show progress bar
        exclude_special_tokens: If True, exclude special tokens like <UNK> from shared vocab.
                               Special tokens are semantically unstable and should not be anchors.

    Returns:
        Tuple of (loaded_models, shared_vocab)
    """
    models = []

    if verbose:
        print(f"Loading {len(model_paths)} models...")
        iterator = tqdm(model_paths, desc="Loading models")
    else:
        iterator = model_paths

    for year, path in iterator:
        model = W2VModel(path)
        # Normalize vectors for stability computation
        model.model.vectors = model.model.vectors.copy()
        model = model.normalize()
        models.append((year, model))

    # Compute shared vocabulary
    if verbose:
        print("Computing shared vocabulary...")

    vocabs = [set(model.vocab) for year, model in models]
    shared_vocab = set.intersection(*vocabs)

    # Filter out special tokens if requested
    if exclude_special_tokens:
        special_tokens = {'<UNK>', '<unk>', '<PAD>', '<pad>', '<S>', '</S>',
                         '<BOS>', '<EOS>', '<MASK>', '<mask>'}
        original_size = len(shared_vocab)
        shared_vocab = {word for word in shared_vocab if word not in special_tokens}

        if verbose and original_size > len(shared_vocab):
            excluded = original_size - len(shared_vocab)
            print(f"Excluded {excluded} special token(s) from shared vocabulary")

    if verbose:
        print(f"Shared vocabulary size: {len(shared_vocab)} words")

    return models, shared_vocab
