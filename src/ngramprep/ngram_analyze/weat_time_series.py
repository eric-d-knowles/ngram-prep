import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from ngramprep.common.w2v_model import W2VModel

def compute_weat_over_years(target1, target2, attribute1, attribute2, start_year, end_year, model_dir, year_step=1, num_permutations=10000, plot=True, smooth=False, sigma=2, confidence=0.95, return_std=True, return_associations=False, custom_combinations=None, plot_associations=None):
    """
    Compute the WEAT effect size, p-value, and error bands over a range of yearly models.

    Args:
        target1 (dict): First set of category words as {'Label': ['word1', 'word2', ...]}.
            Example: {'Male names': ['john', 'paul', 'mike']}
        target2 (dict): Second set of category words. Same format as target1.
        attribute1 (dict): First set of attribute words. Same format as target1.
        attribute2 (dict): Second set of attribute words. Same format as target1.
        start_year (int): The starting year of the range.
        end_year (int): The ending year of the range.
        model_dir (str): Directory containing yearly .kv model files.
        year_step (int): Step size for year increments (default: 1). Should match the year_step used in training.
        num_permutations (int): Number of permutations for significance testing (0 to disable).
        plot (bool or int): If `True`, plots without chunking. If an integer `N`, averages every `N` years for plotting.
        smooth (bool): Whether to overlay a smoothing line over the graph.
        sigma (float): Standard deviation for Gaussian smoothing.
        confidence (float): Confidence level for error bands.
        return_std (bool): Whether to return the standard deviation for error bands.
        return_associations (bool): Whether to return the 4 component mean associations for each year.
        custom_combinations (dict): Optional custom combinations of associations to compute.
            Format: {'Label': ['target1→attribute1', 'target2→attribute2']}
            Example: {'Stereotype-consistent': ['Poor→Unhappy', 'Rich→Happy'],
                     'Stereotype-inconsistent': ['Poor→Happy', 'Rich→Unhappy']}
            The mean of associations in each list will be computed and included in output/plots.
        plot_associations (list): Optional list of specific association keys to plot in the individual associations plot.
            Example: ['Poor→Happy', 'Rich→Unhappy']
            If None, plots all 4 individual associations.

    Returns:
        dict: A dictionary mapping years to tuples. Format depends on parameters:
            - Basic: (effect_size, p_value, std_dev)
            - With return_associations: (effect_size, p_value, std_dev, associations_dict)
              associations_dict includes both individual associations and custom combinations
    """
    # Parse input format (dict with label and words)
    def parse_word_set(word_input, param_name):
        """Extract words and label from input."""
        if not isinstance(word_input, dict):
            raise TypeError(f"{param_name} must be a dictionary with format {{'Label': ['word1', 'word2', ...]}}")
        if len(word_input) != 1:
            raise ValueError(f"{param_name} must contain exactly one key-value pair, got {len(word_input)}")
        label, words = next(iter(word_input.items()))
        return list(words), label

    target1_words, t1_label = parse_word_set(target1, "target1")
    target2_words, t2_label = parse_word_set(target2, "target2")
    attribute1_words, a1_label = parse_word_set(attribute1, "attribute1")
    attribute2_words, a2_label = parse_word_set(attribute2, "attribute2")

    weat_scores = {}
    missing_years = []
    error_years = {}

    for year in range(start_year, end_year + 1, year_step):
        model_pattern = os.path.join(model_dir, f"w2v_y{year}_*.kv")
        model_files = sorted(glob.glob(model_pattern))

        if not model_files:
            missing_years.append(year)
            continue

        model_path = model_files[-1]  # Pick the most recent model

        try:
            yearly_model = W2VModel(model_path)

            # Prepare labels dict if returning associations
            labels = None
            if return_associations:
                labels = {
                    'target1': t1_label,
                    'target2': t2_label,
                    'attribute1': a1_label,
                    'attribute2': a2_label
                }

            weat_result = yearly_model.compute_weat(
                target1_words, target2_words, attribute1_words, attribute2_words, num_permutations,
                return_std=return_std, return_associations=return_associations, labels=labels
            )

            # Parse results based on what was returned
            if return_associations:
                if return_std:
                    effect_size, p_value, std_dev, associations = weat_result
                else:
                    effect_size, p_value, associations = weat_result
                    std_dev = None

                # Compute custom combinations if specified
                if custom_combinations:
                    for combo_label, combo_keys in custom_combinations.items():
                        valid_values = [associations.get(key) for key in combo_keys if associations.get(key) is not None]
                        if valid_values:
                            combo_mean = np.mean(valid_values)
                            associations[combo_label] = combo_mean
                        else:
                            associations[combo_label] = np.nan

                weat_scores[year] = (effect_size, p_value, std_dev, associations)
            else:
                if return_std:
                    effect_size, p_value, std_dev = weat_result
                else:
                    effect_size, p_value = weat_result
                    std_dev = None
                weat_scores[year] = (effect_size, p_value, std_dev)

        except Exception as e:
            error_years[year] = str(e)
            continue

    if missing_years:
        print(f"⚠️ No models found for these years: {missing_years}")

    if error_years:
        print("❌ Errors occurred in the following years:")
        for year, err in error_years.items():
            print(f"  {year}: {err}")

    # Convert results to NumPy arrays for plotting
    if not weat_scores:
        print("❌ No valid WEAT scores computed. Exiting.")
        return {}

    years = np.array(sorted(weat_scores.keys()))
    effect_sizes = np.array([weat_scores[year][0] for year in years])

    # ✅ Handle standard deviations for confidence intervals
    if return_std:
        std_devs = np.array([
            weat_scores[year][2] if weat_scores[year][2] not in [None, 0] else np.nan
            for year in years
        ])
        if np.all(np.isnan(std_devs)):
            print("⚠️ Warning: All standard deviations are NaN. Confidence intervals cannot be plotted.")
            ci_range = np.zeros_like(effect_sizes)  # No error bars
        else:
            ci_range = stats.norm.ppf(1 - (1 - confidence) / 2) * np.nan_to_num(std_devs, nan=np.nanmean(std_devs))
    else:
        ci_range = None  # No confidence intervals if return_std=False

    # ✅ Handle Chunking for Plotting
    chunk_size = plot if isinstance(plot, int) else 1  # Use `plot=N` as chunk size

    if chunk_size > 1:
        chunked_years = []
        chunked_effects = []
        chunked_stds = [] if return_std else None

        for i in range(0, len(years), chunk_size):
            chunk = years[i:i + chunk_size]
            chunk_mean = np.nanmean(effect_sizes[i:i + chunk_size])  # Ignore NaNs
            chunk_year = np.mean(chunk)  # Center of the chunk

            chunked_years.append(chunk_year)
            chunked_effects.append(chunk_mean)

            if return_std:
                chunk_stds = std_devs[i:i + chunk_size]
                chunked_stds.append(np.nanmean(chunk_stds))  # Use mean of std deviations

        years, effect_sizes = np.array(chunked_years), np.array(chunked_effects)
        if return_std:
            std_devs = np.array(chunked_stds)
            ci_range = stats.norm.ppf(1 - (1 - confidence) / 2) * std_devs

    # ✅ Apply Smoothing After Chunking
    smoothed_values = gaussian_filter1d(effect_sizes, sigma=sigma) if smooth else None

    # ✅ Plot Results
    if plot:
        if return_associations:
            # Extract association keys dynamically from the first year's data
            first_year = min(weat_scores.keys())
            all_assoc_keys = list(weat_scores[first_year][3].keys())

            # Separate individual associations from custom combinations
            # Individual associations have the → symbol and are 4 in total
            individual_keys = [k for k in all_assoc_keys if '→' in k and (not custom_combinations or k not in custom_combinations)]
            combo_keys = [k for k in all_assoc_keys if custom_combinations and k in custom_combinations]

            # Filter individual keys based on plot_associations parameter
            if plot_associations is not None:
                individual_keys = [k for k in individual_keys if k in plot_associations]

            # Colors for individual associations
            base_colors = ['green', 'orange', 'purple', 'brown']
            # Colors for custom combinations
            combo_colors = ['red', 'blue', 'magenta', 'cyan', 'black']

            # Plot 1: WEAT Effect Size
            plt.figure(figsize=(10, 5))
            plt.plot(years, effect_sizes, marker='o', linestyle='-', label='WEAT Effect Size', color='blue', linewidth=2)

            if smooth and smoothed_values is not None:
                plt.plot(years, smoothed_values, linestyle='--', color='red', linewidth=2, label='Smoothed Trend')

            if return_std and ci_range is not None:
                plt.fill_between(years, effect_sizes - ci_range, effect_sizes + ci_range, color='blue', alpha=0.2, label=f"{int(confidence * 100)}% CI")

            plt.xlabel("Year", fontsize=12)
            plt.ylabel("WEAT Effect Size (Cohen's d)", fontsize=12)
            plt.title("WEAT Effect Size Over Time", fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Plot 2: Individual Component Associations
            plt.figure(figsize=(10, 5))

            for idx, key in enumerate(individual_keys):
                color = base_colors[idx % len(base_colors)]
                assoc_values = np.array([weat_scores[year][3][key] for year in years])
                plt.plot(years, assoc_values, marker='o', linestyle='-', color=color, label=key, linewidth=1.5, alpha=0.8)

                if smooth:
                    smoothed_assoc = gaussian_filter1d(assoc_values, sigma=sigma)
                    plt.plot(years, smoothed_assoc, linestyle='--', color=color, alpha=0.5, linewidth=2)

            plt.xlabel("Year", fontsize=12)
            plt.ylabel("Mean Cosine Similarity", fontsize=12)
            plt.title("Individual Component Associations Over Time", fontsize=14, fontweight='bold')
            plt.legend(loc='best', fontsize=9)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Plot 3: Custom Combinations (if specified)
            if combo_keys:
                plt.figure(figsize=(10, 5))

                for idx, key in enumerate(combo_keys):
                    color = combo_colors[idx % len(combo_colors)]
                    combo_values = np.array([weat_scores[year][3][key] for year in years])
                    plt.plot(years, combo_values, marker='s', linestyle='-', color=color, label=key, linewidth=2, alpha=0.9)

                    if smooth:
                        smoothed_combo = gaussian_filter1d(combo_values, sigma=sigma)
                        plt.plot(years, smoothed_combo, linestyle='--', color=color, alpha=0.6, linewidth=2)

                plt.xlabel("Year", fontsize=12)
                plt.ylabel("Mean Cosine Similarity", fontsize=12)
                plt.title("Custom Association Combinations Over Time", fontsize=14, fontweight='bold')
                plt.legend(loc='best', fontsize=10)
                plt.grid(True)
                plt.tight_layout()
                plt.show()
        else:
            # Original single plot
            plt.figure(figsize=(10, 5))
            plt.plot(years, effect_sizes, marker='o', linestyle='-', label='WEAT Effect Size', color='blue')

            if smooth and smoothed_values is not None:
                plt.plot(years, smoothed_values, linestyle='--', color='red', label=f'Smoothed Trend')

            if return_std and ci_range is not None:
                plt.fill_between(years, effect_sizes - ci_range, effect_sizes + ci_range, color='blue', alpha=0.2, label=f"{int(confidence * 100)}% CI")

            plt.xlabel("Year")
            plt.ylabel("WEAT Effect Size (Cohen's d)")
            plt.title("WEAT Effect Size Over Time")
            plt.legend()
            plt.grid(True)
            plt.show()

    return weat_scores
