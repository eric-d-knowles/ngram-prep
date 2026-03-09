import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
bin_size = 2  # Set desired bin size (e.g., 2 for year pairs)
use_baseline_corrected_panel = True

# --- Load data ---
# You may need to adjust these paths/names to match your environment
from ipums_utils import fetch_and_aggregate_ipums_professions_csv, calculate_women_percentage

# Example: load results
# result_df = fetch_and_aggregate_ipums_professions_csv(...)
# result = ... # Load projections result dict
# targets = [...] # List of occupation labels

# Pick projection DataFrame (years × professions)
if use_baseline_corrected_panel and result.get('baseline_applied') and not result['projections_corrected'].empty:
    proj_df = result['projections_corrected']
    print("Using baseline-corrected projections")
else:
    proj_df = result['projections']
    print("Using raw projections")

# Get successfully aggregated IPUMS years
ok_years = result_df.loc[result_df['status'] == 'ok'].set_index('year')

# Determine overlapping years (present in both projections and IPUMS)
proj_years = set(proj_df.index)
ipums_years = set(ok_years.index)
panel_years = sorted(proj_years & ipums_years)
print(f"Projection years: {min(proj_df.index)}–{max(proj_df.index)} ({len(proj_df)} total)")
print(f"IPUMS years: {sorted(ipums_years)[:3]}...{sorted(ipums_years)[-3:]}")
print(f"Overlapping years for panel: {len(panel_years)}")

# Assemble panel rows
panel_rows = []
for yr in panel_years:
    csv_path = ok_years.loc[yr, 'output_csv']
    for profession in targets:
        if profession not in proj_df.columns:
            continue
        proj_val = proj_df.loc[yr, profession]
        if pd.isna(proj_val):
            continue
        try:
            wpct = calculate_women_percentage(csv_path, profession)
        except (ZeroDivisionError, ValueError, KeyError):
            continue
        panel_rows.append({
            'profession': profession,
            'year': yr,
            'women_pct': wpct,
            'projection': proj_val,
        })

panel_df = pd.DataFrame(panel_rows)
print(f"\nPanel: {len(panel_df)} observations, "
      f"{panel_df['profession'].nunique()} professions, "
      f"{panel_df['year'].nunique()} years")

# --- YEAR BINNING ---
year_min, year_max = panel_df['year'].min(), panel_df['year'].max()
bin_edges = np.arange(year_min, year_max + 1, bin_size)
panel_df['year_bin'] = panel_df['year'].apply(
    lambda y: bin_edges[np.searchsorted(bin_edges, y, side='right') - 1]
)

panel_binned = (
    panel_df
    .groupby(['profession', 'year_bin'], as_index=False)
    .agg({'women_pct': 'mean', 'projection': 'mean'})
    .rename(columns={'year_bin': 'year'})
)

print(f"Binned panel: {len(panel_binned)} observations, "
      f"{panel_binned['profession'].nunique()} professions, "
      f"{panel_binned['year'].nunique()} bins")

# --- SAVE OUTPUT ---
out_path = 'panel_long_binned.csv'
panel_binned.to_csv(out_path, index=False)
print(f"Saved {out_path}")
