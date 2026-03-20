import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def run_correlation_analysis(panel_df, year_of_interest=None, use_means=False):
    """
    Correlation analysis between women percentage and gender projection scores.

    Parameters
    ----------
    panel_df : DataFrame
        Long-format panel with columns ``profession``, ``year``,
        ``projection``, and ``women_pct``.
    year_of_interest : int or None
        If given (and ``use_means=False``), restrict to a single year.
    use_means : bool
        If True, correlate the over-time mean of each variable per profession.
    """
    if use_means:
        data = panel_df.groupby('profession')[['projection', 'women_pct']].mean().dropna()
        year_label = "all years"
        projection_type = "Mean"
        print(f"Using over-time means for {len(data)} professions")
    else:
        if year_of_interest is None:
            raise ValueError("year_of_interest is required when use_means=False")
        subset = panel_df.loc[panel_df['year'] == year_of_interest]
        if subset.empty:
            available = sorted(panel_df['year'].unique())
            raise ValueError(
                f"No data for year {year_of_interest}. Available: {available}"
            )
        data = subset.set_index('profession')[['projection', 'women_pct']].dropna()
        year_label = str(year_of_interest)
        projection_type = "Raw"
        print(f"Using year {year_of_interest} for {len(data)} professions")

    valid_profs = list(data.index)
    proj_vals = data['projection'].values
    demo_vals = data['women_pct'].values

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
        plt.xlabel(f"IPUMS Women Percentage ({year_label})", fontsize=12, fontweight='bold')
        plt.ylabel(f"Gender Projection Score ({year_label}) - {projection_type}", fontsize=12, fontweight='bold')
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
