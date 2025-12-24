import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from ngramprep.common.w2v_model import W2VModel
from multiprocessing import Pool, cpu_count


def compute_yearly_word_relatedness(args):
    """Helper function for multiprocessing: Computes mean cosine similarity of word(s) to all others."""
    year, model_path, words, excluded_words = args
    try:
        model = W2VModel(model_path)

        # Handle single word or list of words
        if isinstance(words, str):
            words = [words]

        # Compute mean similarity for each word
        similarities = []
        missing_words = []
        for word in words:
            if word not in model.vocab:
                missing_words.append(word)
                continue
            mean_sim = model.mean_cosine_similarity_to_all(word, excluded_words)
            similarities.append(mean_sim)

        if not similarities:
            raise ValueError(f"None of the words {words} found in the model for year {year}.")

        if missing_words:
            print(f"⚠️ Year {year}: Words not found: {missing_words}")

        # Return average similarity across all words
        mean_similarity = np.mean(similarities)
        return (year, mean_similarity, 0)

    except Exception as e:
        return (year, None, str(e))




def track_word_relatedness(
    word, start_year, end_year, model_dir,
    excluded_words=None, year_step=1, plot=True, smooth=False, sigma=2,
    num_workers=None
):
    """
    Track the semantic relatedness/centrality of word(s) over time.

    Computes the yearly mean cosine similarity of word(s) to all other words in the vocabulary.
    This measures how "related" or "central" the word(s) are to the overall vocabulary - higher
    values indicate the word(s) are more semantically connected to other words.

    If multiple words are provided, computes the average relatedness across all specified words.

    Args:
        word (str or list): The target word(s) to track across years. Can be a single word string
                           or a list of words (which will be averaged).
        start_year (int): The starting year of the range.
        end_year (int): The ending year of the range.
        model_dir (str): Directory containing yearly .kv model files.
        excluded_words (list or set): Words to exclude from similarity calculations.
        year_step (int): Step size for year increments (default: 1). Should match the year_step used in training.
        plot (bool or int): If `True`, plots without chunking. If an integer `N`, averages every `N` years.
        smooth (bool): Whether to apply smoothing.
        sigma (float): Standard deviation for Gaussian smoothing.
        num_workers (int or None): Number of parallel workers (default: max CPU cores).

    Returns:
        dict: A dictionary mapping years to (mean cosine similarity, 0).
    """
    similarity_scores = {}
    missing_years = []
    error_years = {}

    # Convert excluded_words to a set for quick lookup
    excluded_words = set(excluded_words) if excluded_words else set()

    # Detect available models
    model_paths = {}
    for year in range(start_year, end_year + 1, year_step):
        model_pattern = os.path.join(model_dir, f"w2v_y{year}_*.kv")
        model_files = sorted(glob.glob(model_pattern))
        if model_files:
            model_paths[year] = model_files[-1]  # Pick the most recent file
        else:
            missing_years.append(year)

    if not model_paths:
        print("❌ No valid models found in the specified range. Exiting.")
        return {}

    # Create label for printing and plotting
    if isinstance(word, str):
        word_label = f"'{word}'"
        words_list = [word]
    else:
        word_label = f"{len(word)} words"
        words_list = word

    print(f"Computing mean cosine similarity for {word_label} (Excluding: {len(excluded_words)} words)")

    # Prepare multiprocessing arguments
    args = [(year, path, word, excluded_words) for year, path in model_paths.items()]

    # Use multiprocessing to compute similarities in parallel
    num_workers = num_workers or min(cpu_count(), len(args))
    with Pool(num_workers) as pool:
        results = pool.map(compute_yearly_word_relatedness, args)

    # Process results
    for year, mean_similarity, std_dev in results:
        if mean_similarity is not None:
            similarity_scores[year] = (mean_similarity, std_dev)
        else:
            error_years[year] = std_dev  # std_dev contains error message

    # Print missing years and errors
    if missing_years:
        print(f"⚠️ No models found for these years: {missing_years}")
    if error_years:
        print("❌ Errors occurred in the following years:")
        for year, err in error_years.items():
            print(f"  {year}: {err}")

    # Convert to NumPy arrays for plotting
    if not similarity_scores:
        print("❌ No valid similarity scores computed. Exiting.")
        return {}

    years = np.array(sorted(similarity_scores.keys()))
    similarities = np.array([similarity_scores[year][0] for year in years])

    # Apply Smoothing
    smoothed_values = gaussian_filter1d(similarities, sigma=sigma) if smooth else None

    # Handle Chunking
    if isinstance(plot, int) and plot > 1:
        chunk_size = plot
        chunked_years = []
        chunked_similarities = []

        for i in range(0, len(years), chunk_size):
            chunk = years[i:i + chunk_size]
            chunk_values = similarities[i:i + chunk_size]

            if len(chunk) > 0:
                chunked_years.append(np.mean(chunk))
                chunked_similarities.append(np.mean(chunk_values))

        years = np.array(chunked_years)
        similarities = np.array(chunked_similarities)

        if smooth:
            smoothed_values = gaussian_filter1d(similarities, sigma=sigma)

    # ✅ Plot Results
    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Create appropriate label based on single word or multiple words
        if isinstance(words_list, list) and len(words_list) == 1:
            plot_label = f"Relatedness of '{words_list[0]}'"
            title = f"Semantic Relatedness of '{words_list[0]}' Over Time"
        else:
            plot_label = f"Avg. Relatedness of {len(words_list)} words"
            title = f"Average Semantic Relatedness Over Time ({len(words_list)} words)"

        ax.scatter(years, similarities, color='blue', alpha=0.2, label=plot_label)
        ax.plot(years, similarities, marker='o', linestyle='-', color='blue', alpha=0.3)

        if smooth and smoothed_values is not None:
            ax.plot(years, smoothed_values, linestyle='--', color='red', linewidth=2, label=f"Smoothed (σ={sigma})")

            # Compute first derivative
            window_length = min(11, len(smoothed_values) if len(smoothed_values) % 2 == 1 else len(smoothed_values) - 1)
            polyorder = min(3, window_length - 1)
            derivative = savgol_filter(smoothed_values, window_length=window_length, polyorder=polyorder, deriv=1, delta=np.mean(np.diff(years)))

            ax2 = ax.twinx()
            ax2.plot(years, derivative, linestyle='-', color='green', linewidth=1, label="First Derivative")
            ax2.set_ylabel("Rate of Change")
            ax2.set_ylim(-0.005, 0.003)

            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
        else:
            ax.legend()

        # Set x-axis limits for consistency with other plots
        ax.set_xlim(start_year - year_step * 0.5, end_year + year_step * 0.5)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean Cosine Similarity to All Vocabulary")
        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    return similarity_scores
