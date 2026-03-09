import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def run_correlation_analysis(year_of_interest, use_baseline_corrected, targets, result_df, result, calculate_women_percentage):
    """
    Correlation analysis between IPUMS women percentage and gender projection scores for a given year.
    Handles NaN values and baseline correction.
    """
    # --- Resolve IPUMS CSV path for the year of interest ---
    ipums_results = result_df
    year_row = ipums_results.loc[
        (ipums_results['year'] == year_of_interest) & (ipums_results['status'] == 'ok')
    ]
    if year_row.empty:
        available = sorted(ipums_results.loc[ipums_results['status'] == 'ok', 'year'].tolist())
        raise ValueError(
            f"No IPUMS CSV for year {year_of_interest}. "
            f"Available years: {available}"
        )
    demo_csv_path = year_row.iloc[0]['output_csv']
    print(f"Using IPUMS data: {demo_csv_path}")

    # Compute women percentages for all targets
    demo_percentages = {}
    for profession in targets:
        try:
            pct = calculate_women_percentage(demo_csv_path, profession)
            demo_percentages[profession] = pct
        except (ZeroDivisionError, ValueError):
            pass

    print(f"Matched {len(demo_percentages)}/{len(targets)} professions in IPUMS data")

    # Determine which projection data to use
    if use_baseline_corrected and result.get('baseline_applied') and not result['projections_corrected'].empty:
        proj_year = result['projections_corrected'].loc[year_of_interest]
        projection_type = "Baseline-corrected"
        print(f"Using baseline-corrected projections")
    else:
        proj_year = result['projections'].loc[year_of_interest]
        projection_type = "Raw"
        print(f"Using raw projections")

    # Create common_profs: professions with both demographic data and projections for this year
    common_profs = [p for p in targets if p in demo_percentages and p in proj_year.index]

    # Filter to professions with valid (non-NaN) projection values
    valid_profs = [p for p in common_profs if not pd.isna(proj_year[p])]

    print(f"Valid professions (non-NaN projections): {len(valid_profs)} out of {len(common_profs)}")
    print(f"Filtered out {len(common_profs) - len(valid_profs)} professions with NaN projections")

    # Extract values for valid professions
    demo_vals = np.array([demo_percentages[p] for p in valid_profs])
    proj_vals = np.array([proj_year[p] for p in valid_profs])

    # Compute correlation
    if len(valid_profs) >= 3:
        corr, p_value = pearsonr(demo_vals, proj_vals)
        print(f"\n=== Correlation Results ({projection_type}) ===")
        print(f"Pearson r = {corr:.4f}")
        print(f"p-value = {p_value:.6f}")
        print(f"n = {len(valid_profs)} professions")

        # Create scatter plot
        plt.figure(figsize=(10, 7))
        plt.scatter(demo_vals, proj_vals, alpha=0.6, s=120, edgecolors='black', linewidth=0.5)
        for i, prof in enumerate(valid_profs):
            plt.annotate(prof, (demo_vals[i], proj_vals[i]), fontsize=9, alpha=0.8, ha='center')
        z = np.polyfit(demo_vals, proj_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(demo_vals.min(), demo_vals.max(), 100)
        plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Linear fit')
        plt.xlabel(f"IPUMS Women Percentage ({year_of_interest})", fontsize=12, fontweight='bold')
        plt.ylabel(f"Gender Projection Score ({year_of_interest}) - {projection_type}", fontsize=12, fontweight='bold')
        plt.title(f"Word Embeddings vs. Real-World Gender Demographics (IPUMS)\nPearson r = {corr:.4f}, p = {p_value:.6f}", fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()
        if p_value < 0.05:
            print(f"\n✓ Significant correlation (p < 0.05)")
        else:
            print(f"\n✗ Not statistically significant (p >= 0.05)")
    else:
        print(f"ERROR: Not enough valid professions ({len(valid_profs)} found)")
