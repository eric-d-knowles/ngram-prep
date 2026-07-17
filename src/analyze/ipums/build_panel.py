import json
import os
import warnings

import numpy as np
import pandas as pd

try:  # package-relative in production; flat import for testing
    from .ipums_utils import calculate_women_proportion
except ImportError:
    from ipums_utils import calculate_women_proportion

# Optional counts helper. Expected signature (mirror of
# calculate_women_proportion, returning the numerator/denominator it
# already computes internally):
#
#     def calculate_women_counts(csv_path, profession, granularity="unigram"):
#         """Return (n_women, n_total) for one profession-year."""
#
# When available, collapse groups use exact count-weighted unions and the
# panel carries women_n / total_n columns (enabling empirical-logit
# measurement models downstream). When absent, build_panel still works:
# collapse falls back to a simple mean of component proportions (with a
# warning) and the count columns are NaN.
try:
    from .ipums_utils import calculate_women_counts
except ImportError:
    try:
        from ipums_utils import calculate_women_counts
    except ImportError:
        calculate_women_counts = None


def _resolve_collapse(targets, collapse):
    """Validate the collapse spec and derive fetch and panel rosters.

    Returns (fetch_professions, panel_roster, component_set).
    - fetch_professions: professions to pull from the IPUMS CSVs
      (non-canonical targets plus every component).
    - panel_roster: unit names in the output panel — targets with
      components replaced (in place, at the first component's position)
      by their canonical name.
    """
    collapse = collapse or {}
    all_components = [c for comps in collapse.values() for c in comps]
    dupes = sorted({c for c in all_components if all_components.count(c) > 1})
    if dupes:
        raise ValueError(f"collapse components appear in multiple groups: {dupes}")
    component_set = set(all_components)
    overlap = component_set & set(collapse.keys()) - {
        canon for canon, comps in collapse.items() if canon in comps}
    if overlap:
        raise ValueError(
            f"canonical name(s) {sorted(overlap)} are components of a "
            f"different group")

    # Professions to fetch from the CSVs: every component, plus targets
    # that aren't canonical labels of a collapse group.
    fetch = [t for t in targets if t not in collapse]
    fetch += [c for c in all_components if c not in fetch]

    # Panel roster: replace each group's components with its canonical,
    # keeping the position of the first component encountered.
    comp_to_canon = {c: canon for canon, comps in collapse.items()
                     for c in comps}
    roster, seen = [], set()
    for t in targets:
        name = comp_to_canon.get(t, t)
        if name not in seen:
            roster.append(name)
            seen.add(name)
    for canon in collapse:            # canonicals not reachable via targets
        if canon not in seen:
            roster.append(canon)
            seen.add(canon)
    return fetch, roster, component_set


def _collapse_demographics(women_df, collapse, component_set):
    """Replace component rows with canonical rows.

    With counts: women_prop(canon, yr) = sum(women_n) / sum(total_n) over
    the components observed that year — the composition of the union.
    This is invariant to duplicated identical sources (k copies of the
    same category give k*w / k*t = w/t), so years where components share
    one census category and years where they are disjoint splits are both
    handled correctly with no seam logic.

    Without counts: simple mean of component proportions (warned).
    """
    if not collapse:
        return women_df
    keep = women_df[~women_df["profession"].isin(component_set)]
    pieces = [keep]
    have_counts = ("total_n" in women_df.columns
                   and women_df["total_n"].notna().any())
    for canon, comps in collapse.items():
        sub = women_df[women_df["profession"].isin(comps)]
        if sub.empty:
            continue
        if have_counts and sub["total_n"].notna().all():
            g = (sub.groupby("year", as_index=False)
                 .agg(women_n=("women_n", "sum"),
                      total_n=("total_n", "sum")))
            g["women_prop"] = g["women_n"] / g["total_n"]
        else:
            warnings.warn(
                f"collapse group '{canon}': counts unavailable — using a "
                f"simple mean of component proportions. Add "
                f"calculate_women_counts to ipums_utils for exact "
                f"employment-weighted unions.", UserWarning)
            g = (sub.groupby("year", as_index=False)
                 .agg(women_prop=("women_prop", "mean")))
            g["women_n"] = np.nan
            g["total_n"] = np.nan
        g["profession"] = canon
        pieces.append(g[["profession", "year", "women_prop",
                         "women_n", "total_n"]])
    return pd.concat(pieces, ignore_index=True)


def _projection_series(proj_df, name, collapse):
    """Projection series for a panel unit.

    For a collapse canonical: per-year mean over whichever component
    columns (and the canonical column itself, if the upstream synonyms
    mechanism already produced one) exist in proj_df. For an ordinary
    unit: its own column.
    """
    if collapse and name in collapse:
        cols = [c for c in dict.fromkeys([*collapse[name], name])
                if c in proj_df.columns]
        if not cols:
            return None
        return proj_df[cols].mean(axis=1)
    if name in proj_df.columns:
        return proj_df[name]
    return None


def build_panel(
    projection_dict,
    cps_manifest,
    targets,
    use_baseline_corrected_panel=True,
    bin_size=1,
    granularity="unigram",
    cache_file=None,
    collapse=None,
):
    """
    Build long-format panel: (profession, year, women_prop, women_n,
    total_n, projection) with optional year binning and optional
    collapsing of multiple professions into one canonical unit.

    collapse : dict[str, list[str]] or None
        Mapping of canonical unit name -> list of component professions,
        e.g. ``{"examiner": ["appraiser", "examiner", "investigator"]}``.
        Components are removed from the panel and replaced by the
        canonical unit, whose:
          * women_prop is the count-weighted union of the components
            (sum of women / sum of totals per year — exact composition
            of the merged occupation; requires calculate_women_counts in
            ipums_utils, else falls back to a simple mean with a
            warning). The ratio-of-sums is invariant to years where the
            components share a single census category, so pre/post
            classification-split periods need no special handling.
          * projection is the per-year mean of the component columns in
            the projection DataFrame (plus the canonical column if the
            upstream ``synonyms`` mechanism already created one). Do not
            ALSO list the same group under ``synonyms`` upstream unless
            you want the canonical column included in this mean — either
            layer alone is sufficient and they compose harmlessly.

    cache_file : str, None, or False
        Path to a parquet cache for the computed women-share rows.
        None  (default) – auto-derive as
                          ``women_prop_{granularity}_v2.parquet`` in the
                          same directory as the source CSVs.
        False           – disable caching entirely.
        str             – use the given path.
        The cache stores PER-PROFESSION rows (components, not collapsed
        units) including counts, so it is valid across different
        ``collapse`` specs. It is invalidated automatically when any
        source CSV is newer than the cache file, or when professions are
        requested that the cache never attempted. (The filename gains a
        ``_v2`` suffix because the schema now includes count columns —
        old caches are simply ignored.)

    Returns:
        panel_df: DataFrame with columns
        [profession, year, women_prop, women_n, total_n, projection].
        Count columns are NaN when calculate_women_counts is unavailable
        or a collapse group fell back to the simple mean.
    """
    # Pick projection DataFrame
    if (use_baseline_corrected_panel
            and projection_dict.get("baseline_applied")
            and not projection_dict["projections_corrected"].empty):
        proj_df = projection_dict["projections_corrected"]
    else:
        proj_df = projection_dict["projections"]

    fetch_professions, roster, component_set = _resolve_collapse(
        targets, collapse)

    # Get all consecutive IPUMS years for women share
    ok_years = cps_manifest.loc[cps_manifest["status"] == "ok"].set_index("year")
    ipums_years = sorted(ok_years.index)
    csv_paths = [
        ok_years.loc[yr, "output_csv"] for yr in ipums_years
        if os.path.exists(ok_years.loc[yr, "output_csv"])
    ]

    # Auto-derive cache path from the data directory (v2: schema + counts)
    if cache_file is None and csv_paths:
        cache_file = os.path.join(
            os.path.dirname(csv_paths[0]),
            f"women_prop_{granularity}_v2.parquet",
        )

    # Attempt cache load (cache holds per-profession rows incl. counts)
    _meta = cache_file.replace(".parquet", ".attempted.json") if cache_file else None
    women_df = None
    if cache_file and os.path.exists(cache_file) and _meta and os.path.exists(_meta):
        cache_mtime = os.path.getmtime(cache_file)
        sources_fresh = all(os.path.getmtime(p) <= cache_mtime for p in csv_paths)
        if sources_fresh:
            with open(_meta) as _f:
                attempted = set(json.load(_f))
            if set(fetch_professions).issubset(attempted):
                cached = pd.read_parquet(cache_file)
                women_df = cached[
                    cached["profession"].isin(fetch_professions)].copy()

    # Compute if no valid cache
    if women_df is None:
        women_rows = []
        for yr in ipums_years:
            csv_path = ok_years.loc[yr, "output_csv"]
            for profession in fetch_professions:
                w_n = t_n = np.nan
                if calculate_women_counts is not None:
                    try:
                        w_n, t_n = calculate_women_counts(
                            csv_path, profession, granularity=granularity)
                        if not t_n or t_n <= 0:
                            continue
                        wprop = w_n / t_n
                    except (ZeroDivisionError, ValueError, KeyError):
                        continue
                else:
                    try:
                        wprop = calculate_women_proportion(
                            csv_path, profession, granularity=granularity)
                    except (ZeroDivisionError, ValueError, KeyError):
                        continue
                women_rows.append({
                    "profession": profession,
                    "year": yr,
                    "women_prop": wprop,
                    "women_n": w_n,
                    "total_n": t_n,
                })
        women_df = pd.DataFrame(
            women_rows,
            columns=["profession", "year", "women_prop", "women_n", "total_n"])
        if cache_file and not women_df.empty:
            os.makedirs(os.path.dirname(os.path.abspath(cache_file)),
                        exist_ok=True)
            women_df.to_parquet(cache_file, index=False)
            # Record all attempted professions (including those with no data)
            with open(_meta, "w") as _f:
                json.dump(sorted(fetch_professions), _f)

    # Collapse component professions into canonical units (yearly level,
    # counts-weighted) BEFORE binning.
    women_df = _collapse_demographics(women_df, collapse, component_set)

    # Bin women share by year_bin
    if women_df.empty:
        return pd.DataFrame(columns=["profession", "year", "women_prop",
                                     "women_n", "total_n", "projection"])
    if bin_size > 1:
        base_year = min(proj_df.index)  # or hardcode 1968 if that's always the start
        women_df["year_bin"] = ((women_df["year"] - base_year)
                                // bin_size) * bin_size + base_year
    else:
        women_df["year_bin"] = women_df["year"]

    # Bin aggregation: proportions by mean (legacy behavior), counts by
    # sum so the count columns remain meaningful at bin level.
    women_binned = (
        women_df.groupby(["profession", "year_bin"], as_index=False)
        .agg(women_prop=("women_prop", "mean"),
             women_n=("women_n", lambda s: s.sum(min_count=1)),
             total_n=("total_n", lambda s: s.sum(min_count=1)))
        .rename(columns={"year_bin": "year"})
    )

    # Assemble panel rows over the roster (canonicals, not components)
    proj_years = set(proj_df.index)
    panel_years = sorted(set(women_binned["year"]) & proj_years)

    panel_rows = []
    for yr in panel_years:
        for name in roster:
            proj_series = _projection_series(proj_df, name, collapse)
            if proj_series is None:
                continue
            proj_val = proj_series.loc[yr] if yr in proj_series.index else np.nan
            if pd.isna(proj_val):
                continue
            wrow = women_binned.loc[
                (women_binned["profession"] == name)
                & (women_binned["year"] == yr)]
            if wrow.empty:
                continue
            panel_rows.append({
                "profession": name,
                "year": yr,
                "women_prop": wrow["women_prop"].iloc[0],
                "women_n": wrow["women_n"].iloc[0],
                "total_n": wrow["total_n"].iloc[0],
                "projection": proj_val,
            })

    panel_df = pd.DataFrame(
        panel_rows,
        columns=["profession", "year", "women_prop", "women_n", "total_n",
                 "projection"])
    return panel_df
