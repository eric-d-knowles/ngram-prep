import pandas as pd
import numpy as np

def build_panel(
    result,
    result_df,
    targets,
    calculate_women_percentage,
    use_baseline_corrected_panel=True,
    bin_size=1
):
    """
    Build long-format panel: (profession, year, women_pct, projection)
    with optional year binning.

    Returns:
        panel_df: DataFrame with columns [profession, year, women_pct, projection]
    """
    # Pick projection DataFrame
    if use_baseline_corrected_panel and result.get('baseline_applied') and not result['projections_corrected'].empty:
        proj_df = result['projections_corrected']
    else:
        proj_df = result['projections']

    # Get all consecutive IPUMS years for women%
    ok_years = result_df.loc[result_df['status'] == 'ok'].set_index('year')
    ipums_years = sorted(ok_years.index)

    # Build women% panel for all IPUMS years
    women_rows = []
    for yr in ipums_years:
        csv_path = ok_years.loc[yr, 'output_csv']
        for profession in targets:
            try:
                wpct = calculate_women_percentage(csv_path, profession)
            except (ZeroDivisionError, ValueError, KeyError):
                continue
            women_rows.append({
                'profession': profession,
                'year': yr,
                'women_pct': wpct,
            })
    women_df = pd.DataFrame(women_rows)

    # Bin women% by year_bin
    if bin_size > 1 and not women_df.empty:
        base_year = min(proj_df.index)  # or hardcode 1968 if that's always the start
        women_df['year_bin'] = ((women_df['year'] - base_year) // bin_size) * bin_size + base_year
    else:
        women_df['year_bin'] = women_df['year']

    # Compute mean women% for each bin and profession
    women_binned = (
        women_df.groupby(['profession', 'year_bin'], as_index=False)
        .agg({'women_pct': 'mean'})
        .rename(columns={'year_bin': 'year'})
    )

    # Prepare projections panel (already binned)
    proj_years = set(proj_df.index)
    panel_years = sorted(set(women_binned['year']) & proj_years)

    panel_rows = []
    for yr in panel_years:
        for profession in targets:
            if profession not in proj_df.columns:
                continue
            proj_val = proj_df.loc[yr, profession]
            if pd.isna(proj_val):
                continue
            wpct_row = women_binned.loc[
                (women_binned['profession'] == profession) & (women_binned['year'] == yr),
                'women_pct'
            ]
            if wpct_row.empty:
                continue
            panel_rows.append({
                'profession': profession,
                'year': yr,
                'women_pct': wpct_row.iloc[0],
                'projection': proj_val,
            })

    panel_df = pd.DataFrame(panel_rows)
    return panel_df