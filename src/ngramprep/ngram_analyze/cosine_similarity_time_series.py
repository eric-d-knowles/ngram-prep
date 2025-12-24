import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from gensim.models import KeyedVectors

def cosine_similarity_over_years(word1, word2, start_year, end_year, model_dir, year_step=1, plot=True, smooth=False, sigma=2):
    """
    Compute the cosine similarity between two words (or word groups) across a range of yearly models.

    If lists are provided for word1 and/or word2, computes all pairwise similarities and returns
    the average similarity.

    Args:
        word1 (str or list): The first word or list of words.
        word2 (str or list): The second word or list of words.
        start_year (int): The starting year of the range.
        end_year (int): The ending year of the range.
        model_dir (str): Directory containing yearly .kv model files.
        year_step (int): Step size for year increments (default: 1). Should match the year_step used in training.
        plot (bool or int): If `True`, plots yearly data. If an integer `N`, averages every `N` years for plotting.
        smooth (bool): Whether to overlay a smoothing line on the graph.
        sigma (float): Standard deviation for Gaussian smoothing (higher = smoother curve).

    Returns:
        dict: A dictionary mapping years to cosine similarity scores.
    """
    if not os.path.exists(model_dir):
        print(f"Model directory '{model_dir}' does not exist. Please check the path.")
        return {}

    # Convert to lists if needed
    words1 = [word1] if isinstance(word1, str) else word1
    words2 = [word2] if isinstance(word2, str) else word2

    similarities = {}
    missing_models = []

    for year in range(start_year, end_year + 1, year_step):
        model_pattern = os.path.join(model_dir, f"w2v_y{year}_*.kv")
        model_files = glob.glob(model_pattern)

        if not model_files:
            missing_models.append(year)
            continue  # Skip missing models

        model_path = model_files[0]  # Pick the first matching model

        try:
            yearly_model = KeyedVectors.load(model_path, mmap="r")

            # Compute all pairwise similarities
            pairwise_sims = []
            missing_words = []

            for w1 in words1:
                for w2 in words2:
                    if w1 in yearly_model.key_to_index and w2 in yearly_model.key_to_index:
                        sim = yearly_model.similarity(w1, w2)
                        pairwise_sims.append(sim)
                    else:
                        if w1 not in yearly_model.key_to_index:
                            missing_words.append(w1)
                        if w2 not in yearly_model.key_to_index:
                            missing_words.append(w2)

            if pairwise_sims:
                # Average all pairwise similarities
                similarities[year] = np.mean(pairwise_sims)
                if missing_words:
                    print(f"⚠️ Year {year}: Some words not found: {set(missing_words)}")
            else:
                print(f"⚠️ Year {year}: No valid word pairs found")

        except Exception as e:
            print(f"Skipping {year} due to error: {e}")
            continue

    if not similarities:
        print("❌ No valid similarity scores computed. Exiting.")
        return {}

    # ✅ Create range of years based on year_step
    all_years = np.arange(start_year, end_year + 1, year_step)
    similarity_values = np.array([similarities.get(year, np.nan) for year in all_years])

    # ✅ Interpolate missing values (only for gaps in the stepped years)
    mask = ~np.isnan(similarity_values)
    if mask.any() and not mask.all():
        similarity_values = np.interp(all_years, all_years[mask], similarity_values[mask])

    # ✅ Set `chunk_size` Based on `plot`
    chunk_size = plot if isinstance(plot, int) else 1  # If `plot=N`, set `chunk_size=N`

    # ✅ Apply Chunking (Averaging Consecutive Years)
    if chunk_size > 1:
        chunked_years = []
        chunked_similarities = []

        for i in range(0, len(all_years), chunk_size):
            chunk = all_years[i:i + chunk_size]
            chunk_values = similarity_values[i:i + chunk_size]

            if np.isnan(chunk_values).all():
                continue  # Skip chunks with only missing values

            chunk_mean = np.nanmean(chunk_values)  # Ignore NaNs in averaging
            chunk_year = np.mean(chunk)  # Center of the chunk

            chunked_years.append(chunk_year)
            chunked_similarities.append(chunk_mean)

        # ✅ Replace full-year data with chunked data
        all_years, similarity_values = np.array(chunked_years), np.array(chunked_similarities)

    # ✅ Apply Smoothing (After Chunking)
    smoothed_values = gaussian_filter1d(similarity_values, sigma=sigma) if smooth else None

    # ✅ Plot the Results
    if plot:
        # Create appropriate labels
        if isinstance(words1, list) and len(words1) == 1:
            label1 = f"'{words1[0]}'"
        else:
            label1 = f"{len(words1)} words"

        if isinstance(words2, list) and len(words2) == 1:
            label2 = f"'{words2[0]}'"
        else:
            label2 = f"{len(words2)} words"

        plot_label = f"Similarity ({label1}, {label2})"
        title = f"Cosine Similarity: {label1} and {label2} Over Time"

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(all_years, similarity_values, color='blue', alpha=0.2, label=plot_label)
        ax.plot(all_years, similarity_values, marker='o', linestyle='-', color='blue', alpha=0.3)

        if smooth and smoothed_values is not None:
            ax.plot(all_years, smoothed_values, linestyle='--', color='red', linewidth=2, label=f'Smoothed (σ={sigma})')

            # Compute first derivative
            window_length = min(11, len(smoothed_values) if len(smoothed_values) % 2 == 1 else len(smoothed_values) - 1)
            polyorder = min(3, window_length - 1)
            derivative = savgol_filter(smoothed_values, window_length=window_length, polyorder=polyorder, deriv=1, delta=np.mean(np.diff(all_years)))

            ax2 = ax.twinx()
            ax2.plot(all_years, derivative, linestyle='-', color='green', linewidth=1, label="First Derivative")
            ax2.set_ylabel("Rate of Change")
            ax2.set_ylim(-0.005, 0.003)

            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
        else:
            ax.legend()

        # Set x-axis limits for consistency with other plots
        ax.set_xlim(start_year - year_step * 0.5, end_year + year_step * 0.5)
        ax.set_xlabel("Year")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    return similarities


def plot_nearest_neighbors(word, year, model_dir, n=10, figsize=(10, 6)):
    """
    Plot the N nearest neighbors to a target word in a specific year.

    Args:
        word (str): The target word to find neighbors for.
        year (int): The year of the corpus to use.
        model_dir (str): Directory containing yearly .kv model files.
        n (int): Number of nearest neighbors to show (default: 10).
        figsize (tuple): Figure size for the plot (default: (10, 6)).

    Returns:
        list: List of (word, similarity) tuples for the nearest neighbors,
              or None if the word is not in the vocabulary.
    """
    if not os.path.exists(model_dir):
        print(f"Model directory '{model_dir}' does not exist. Please check the path.")
        return None

    # Find the model file for the specified year
    model_pattern = os.path.join(model_dir, f"w2v_y{year}_*.kv")
    model_files = glob.glob(model_pattern)

    if not model_files:
        print(f"⚠️ Warning: No model file found for year {year}")
        return None

    model_path = model_files[0]

    try:
        model = KeyedVectors.load(model_path, mmap="r")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    # Check if word is in vocabulary
    if word not in model.key_to_index:
        print(f"⚠️ Warning: '{word}' not found in {year} model vocabulary")
        return None

    # Get nearest neighbors
    try:
        neighbors = model.most_similar(word, topn=n)
    except Exception as e:
        print(f"❌ Error computing neighbors: {e}")
        return None

    # Extract words and similarities
    words, similarities = zip(*neighbors)

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(words))

    # Plot bars in reverse order so highest similarity is at top
    ax.barh(y_pos, similarities[::-1], align='center', color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words[::-1])
    ax.invert_yaxis()  # Highest at top
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_title(f'Top {n} Nearest Neighbors to "{word}" ({year})', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])

    # Add value labels on bars
    for i, v in enumerate(similarities[::-1]):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    return neighbors
