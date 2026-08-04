"""
build_panel v4 — v3 plus Fem-side observation-error pass-through.

v4 change (July 2026, Fem-side injection):

  * FEM_VAR PASS-THROUGH. When the projection dict carries the "fem_var"
    frame emitted by the FEMVAR-patched compute_projection_over_years
    (per-year estimation-variance proxies from the yearly models' own
    type counts), the panel gains a fem_var column. Ordinary units read
    their column directly; collapse canonicals whose projection is
    averaged HERE (components present as separate proj_df columns, i.e.
    the group was NOT handled by upstream ``synonyms``) get the variance
    of that mean, (1/k²)·Σ fem_var over exactly the columns
    _projection_series averages, with the same per-year availability and
    the same strictness rule as upstream (any averaged column missing
    fem_var ⇒ NaN, never a silent partial sum). When the canonical
    column was produced upstream, its fem_var is already
    availability-matched there and passes straight through.

    The column is OMITTED (with a printed note) when the projection dict
    has no usable fem_var — so augment_panel_with_noise's panel-native
    check falls back cleanly instead of seeing an all-NaN column.

    CACHE: unaffected. The women_prop_*_v3.parquet caches DEMOGRAPHIC
    count rows only; fem_var rides the projection dict, which build_panel
    never caches. No cache-version bump — but any notebook-level panel
    or projection-dict parquets saved before the column existed are
    stale by construction.

    Baseline correction: fem_var lookups are independent of whether
    projections_corrected was selected — subtracting a common yearly
    baseline does not change per-unit estimation variance, and the
    corrected frame preserves the raw frame's NaN pattern.

──────────────────────────────────────────────────────────────────────────
build_panel v3 — counts-native panel assembly (v2 processed files only).

Changes vs v2 (July 2026 measurement-error prerequisite work):

  * UNWEIGHTED COUNTS. The panel carries women_n_unw / total_n_unw
    (raw CPS respondent counts) alongside the weighted women_n / total_n.
    The unweighted pair is the honest (x, n) for the empirical logit
    log((x+0.5)/(n-x+0.5)) and its known sampling variance
    Var ~= 1/(x+0.5) + 1/(n-x+0.5).

  * DUPLICATION-SAFE COLLAPSE. Collapse canonicals are computed directly
    in the fetch loop via a label-union row mask (ipums_utils.
    calculate_year_counts), so each processed row is counted once even in
    years where components share a single census category. The old
    sum-of-component-counts path was ratio-invariant (k*w / k*t) but
    inflated n by the duplication factor — understating empirical-logit
    variance 3x for the examiner triad's shared years. _collapse_demographics
    is deleted; there is nothing left to collapse after fetch.

  * ONE READ PER YEAR. calculate_year_counts matches every unit against a
    single in-memory frame per year's CSV (58 reads total, down from
    thousands of per-(unit, year) reads).

  * v3 CACHE with a spec fingerprint. The parquet stores UNIT-level rows
    (canonicals included), so cache validity now depends on the collapse
    spec: the .attempted.json sidecar records each unit's exact label
    list, and the cache is used only when every requested unit appears
    with an identical label list. mtime invalidation against the source
    CSVs is unchanged; pointing at the regenerated professions_v2
    directory yields a fresh cache automatically.

  * NO FALLBACKS. v2 counts-schema files are required; ipums_utils'
    counts helpers are hard imports. Missing capability fails loudly.

  * BIN ESTIMATOR. With counts always present, bins aggregate by
    ratio-of-summed-counts (weighted for women_prop; unweighted summed
    for the variance path) instead of averaging proportions. Identical
    at bin_size=1 (the production setting); at bin_size>1 this is the
    employment-weighted bin composition, which is the estimator you want.

Panel schema:
    [profession, year, women_prop, women_n, total_n,
     women_n_unw, total_n_unw, projection, fem_var?]
    (fem_var present only when the projection dict supplies it)
"""

import json
import os

import numpy as np
import pandas as pd

try:  # package-relative in production; flat import for testing
    from .ipums_utils import calculate_year_counts
except ImportError:
    from ipums_utils import calculate_year_counts


def _resolve_collapse(targets, collapse):
    """Validate the collapse spec and derive the unit map and panel roster.

    Returns (units, roster):
      - units: dict mapping panel unit name -> list of labels defining it.
        Ordinary targets map to themselves ({"nurse": ["nurse"]});
        collapse canonicals map to their component list.
      - roster: unit names in panel order — targets with components
        replaced (in place, at the first component's position) by their
        canonical name; canonicals unreachable via targets appended.
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

    units = {}
    for name in roster:
        if name in collapse:
            units[name] = list(dict.fromkeys(collapse[name]))
        else:
            units[name] = [name]
    return units, roster


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


# FEMVAR: edit P1 — fem_var series mirroring _projection_series exactly.
def _fem_var_series(fem_var_df, proj_df, name, collapse):
    """fem_var series for a panel unit, matched to _projection_series.

    Ordinary unit: its fem_var column (already availability-matched
    upstream). Collapse canonical averaged HERE: the variance of the
    panel-level mean — (1/k²)·Σ fem_var over EXACTLY the columns
    _projection_series averages, with k the per-year count of non-NaN
    projection columns and the strictness rule that any averaged column
    missing fem_var yields NaN for that year (no silent partial sums).
    Returns None when nothing is computable for the unit.
    """
    if fem_var_df is None or fem_var_df.empty:
        return None
    if collapse and name in collapse:
        cols = [c for c in dict.fromkeys([*collapse[name], name])
                if c in proj_df.columns]
        if not cols:
            return None
        avail = proj_df[cols].notna()
        k = avail.sum(axis=1)
        fv_cols = fem_var_df.reindex(columns=cols)
        masked = fv_cols.where(avail)
        n_counted = masked.notna().sum(axis=1)
        ssum = masked.sum(axis=1)
        fv = ssum / (k.astype(float) ** 2)
        fv[(k == 0) | (n_counted != k)] = np.nan
        return fv
    if name in fem_var_df.columns:
        return fem_var_df[name]
    return None


def _cache_paths(cache_file):
    meta = cache_file.replace(".parquet", ".attempted.json")
    return cache_file, meta


def _cache_valid(cache_file, meta_file, csv_paths, units):
    """Cache usable iff sources are not newer AND every requested unit was
    attempted with an identical label list (collapse-spec fingerprint)."""
    if not (os.path.exists(cache_file) and os.path.exists(meta_file)):
        return False
    cache_mtime = os.path.getmtime(cache_file)
    if any(os.path.getmtime(p) > cache_mtime for p in csv_paths):
        return False
    with open(meta_file) as f:
        meta = json.load(f)
    attempted = meta.get("units", {})
    for name, labels in units.items():
        if attempted.get(name) != sorted(labels):
            return False
    return True


def build_panel(
    projection_dict,
    cps_manifest,
    targets,
    use_baseline_corrected_panel=True,
    bin_size=1,
    granularity="unigram",
    cache_file=None,
    collapse=None,
    require_projection=True,
):
    """
    Build the long-format panel:
    (profession, year, women_prop, women_n, total_n, women_n_unw,
    total_n_unw, projection[, fem_var]), with optional year binning and
    optional collapsing of multiple professions into one canonical unit.

    require_projection : bool, default True
        True (legacy): a unit-year is emitted only when a projection
        exists for it, so the panel is complete-case on the projection
        side and ``panel_years`` is the intersection of demographic and
        projection years.

        False: the DEMOGRAPHIC grid drives the panel and projections join
        where available, emitting ``projection = NaN`` (and ``fem_var =
        NaN``) otherwise.  This is the ASYMMETRIC-BIN case: pass
        ``bin_size=1`` with a projection dict built on coarser bins
        (e.g. 2-year Fiction models) and you keep every annual W%
        observation while Fem appears only on its bin-label years.
        Downstream, ctsem's Kalman filter handles the NaN manifests
        natively (``complete_only=False``), and the calendar-anchored
        time axis keeps the gaps as real dt.  Units with NO projection
        column at all are still dropped — this admits gaps WITHIN a
        unit's series, not units without any Fem data.

    collapse : dict[str, list[str]] or None
        Mapping of canonical unit name -> list of component professions,
        e.g. ``{"examiner": ["appraiser", "examiner", "investigator"]}``.
        Components are removed from the panel and replaced by the
        canonical unit. Demographics for a canonical are the
        duplication-safe union of every processed row matching ANY
        component label (each row counted once — correct for both the
        composition ratio and the respondent counts, in shared-category
        and split years alike). Its projection is the per-year mean of
        the component columns in the projection DataFrame (plus the
        canonical column if the upstream ``synonyms`` mechanism already
        created one). Do not ALSO list the same group under ``synonyms``
        upstream unless you want the canonical column included in this
        mean — either layer alone is sufficient and they compose
        harmlessly. When fem_var is present, a panel-level mean carries
        the matching (1/k²)·Σ variance over the same columns.

    granularity : str
        Retained for signature compatibility and cache naming; the v2
        counts helpers do not use it.

    cache_file : str, None, or False
        Path to a parquet cache for the computed count rows.
        None  (default) – auto-derive as
                          ``women_prop_{granularity}_v3.parquet`` in the
                          same directory as the source CSVs.
        False           – disable caching entirely.
        str             – use the given path.
        The v3 cache stores UNIT-level rows (canonicals computed
        directly), so its sidecar records each unit's exact label list;
        the cache is bypassed when any requested unit's labels differ
        (collapse-spec change), when any source CSV is newer, or when a
        requested unit was never attempted. fem_var never enters this
        cache (it rides the projection dict), so v4 needs no cache bump.

    Returns:
        panel_df with columns [profession, year, women_prop, women_n,
        total_n, women_n_unw, total_n_unw, projection] plus fem_var when
        the projection dict supplies a non-empty "fem_var" frame (the
        FEMVAR-patched compute_projection_over_years). Weighted counts
        (women_n/total_n) are the estimator inputs; unweighted counts
        (women_n_unw/total_n_unw) and fem_var are the measurement-error
        inputs.
    """
    # Pick projection DataFrame
    if (use_baseline_corrected_panel
            and projection_dict.get("baseline_applied")
            and not projection_dict["projections_corrected"].empty):
        proj_df = projection_dict["projections_corrected"]
    else:
        proj_df = projection_dict["projections"]

    # FEMVAR: edit P2 — pull the fem_var frame (independent of the
    # corrected/raw choice above; see module docstring).
    fem_var_df = projection_dict.get("fem_var")
    has_fem_var = (isinstance(fem_var_df, pd.DataFrame)
                   and not fem_var_df.empty)
    if not has_fem_var:
        fem_var_df = None
        print("build_panel: projection dict has no usable 'fem_var' frame — "
              "panel will omit the fem_var column here. Fem-side noise can "
              "still be added downstream via augment_panel_with_noise("
              "replica_cube=... for noise-ensemble replicas, or counts_df="
              "...+groups=... for token counts), or upstream by passing a "
              "FEMVAR-patched compute_projection_over_years result instead.")

    units, roster = _resolve_collapse(targets, collapse)

    # All usable IPUMS years from the manifest
    ok_years = cps_manifest.loc[cps_manifest["status"] == "ok"].set_index("year")
    ipums_years = sorted(ok_years.index)
    csv_paths = [
        ok_years.loc[yr, "output_csv"] for yr in ipums_years
        if os.path.exists(ok_years.loc[yr, "output_csv"])
    ]

    # Auto-derive cache path (v3: unit-level rows + unweighted counts)
    if cache_file is None and csv_paths:
        cache_file = os.path.join(
            os.path.dirname(csv_paths[0]),
            f"women_prop_{granularity}_v3.parquet",
        )

    women_df = None
    meta_file = None
    if cache_file:
        cache_file, meta_file = _cache_paths(cache_file)
        if _cache_valid(cache_file, meta_file, csv_paths, units):
            cached = pd.read_parquet(cache_file)
            women_df = cached[cached["profession"].isin(units)].copy()

    # Compute if no valid cache: ONE read per year, all units at once
    if women_df is None:
        pieces = []
        for yr in ipums_years:
            csv_path = ok_years.loc[yr, "output_csv"]
            if not os.path.exists(csv_path):
                continue
            year_counts = calculate_year_counts(csv_path, units)
            if year_counts.empty:
                continue
            year_counts["year"] = yr
            pieces.append(year_counts)
        if pieces:
            women_df = pd.concat(pieces, ignore_index=True)
        else:
            women_df = pd.DataFrame(
                columns=["profession", "women_n", "total_n",
                         "women_n_unw", "total_n_unw", "year"])
        women_df["women_prop"] = women_df["women_n"] / women_df["total_n"]

        if cache_file and not women_df.empty:
            os.makedirs(os.path.dirname(os.path.abspath(cache_file)),
                        exist_ok=True)
            women_df.to_parquet(cache_file, index=False)
            with open(meta_file, "w") as f:
                json.dump(
                    {"units": {n: sorted(ls) for n, ls in units.items()}},
                    f, indent=1)

    if women_df.empty:
        base_cols = ["profession", "year", "women_prop", "women_n",
                     "total_n", "women_n_unw", "total_n_unw", "projection"]
        return pd.DataFrame(
            columns=base_cols + (["fem_var"] if has_fem_var else []))

    # ── Bin by year_bin (ratio-of-summed-counts estimator) ───────────────
    if bin_size > 1:
        base_year = min(proj_df.index)
        women_df = women_df.copy()
        women_df["year_bin"] = ((women_df["year"] - base_year)
                                // bin_size) * bin_size + base_year
    else:
        women_df = women_df.copy()
        women_df["year_bin"] = women_df["year"]

    women_binned = (
        women_df.groupby(["profession", "year_bin"], as_index=False)
        .agg(women_n=("women_n", "sum"),
             total_n=("total_n", "sum"),
             women_n_unw=("women_n_unw", "sum"),
             total_n_unw=("total_n_unw", "sum"))
        .rename(columns={"year_bin": "year"})
    )
    women_binned["women_prop"] = (
        women_binned["women_n"] / women_binned["total_n"])

    # ── Assemble panel rows over the roster ──────────────────────────────
    proj_years = set(proj_df.index)
    if require_projection:
        panel_years = sorted(set(women_binned["year"]) & proj_years)
    else:
        # Demographic grid drives the panel; projections join where
        # available.  Fem-missing occasions become NaN rather than
        # disappearing, so a coarser projection grid does not throw away
        # the finer W% series.
        panel_years = sorted(set(women_binned["year"]))

    # Resolve each unit's series ONCE.  These were previously recomputed
    # inside the year loop, i.e. len(panel_years) times per unit.  Units
    # with no projection column are dropped here as before — but now
    # reported rather than vanishing silently.
    unit_proj, unit_fv, no_proj_col = {}, {}, []
    for name in roster:
        ps = _projection_series(proj_df, name, collapse)
        if ps is None:
            no_proj_col.append(name)
            continue
        unit_proj[name] = ps
        unit_fv[name] = (_fem_var_series(fem_var_df, proj_df, name, collapse)
                         if has_fem_var else None)
    if no_proj_col:
        print(f"build_panel: {len(no_proj_col)} of {len(roster)} units have "
              f"no projection column and are absent from the panel: "
              f"{', '.join(map(str, no_proj_col))}")

    panel_rows = []
    for yr in panel_years:
        for name in roster:
            if name not in unit_proj:
                continue
            proj_series = unit_proj[name]
            proj_val = (proj_series.loc[yr]
                        if yr in proj_series.index else np.nan)
            if require_projection and pd.isna(proj_val):
                continue
            wrow = women_binned.loc[
                (women_binned["profession"] == name)
                & (women_binned["year"] == yr)]
            if wrow.empty:
                continue
            row = {
                "profession": name,
                "year": yr,
                "women_prop": wrow["women_prop"].iloc[0],
                "women_n": wrow["women_n"].iloc[0],
                "total_n": wrow["total_n"].iloc[0],
                "women_n_unw": int(wrow["women_n_unw"].iloc[0]),
                "total_n_unw": int(wrow["total_n_unw"].iloc[0]),
                "projection": proj_val,
            }
            # FEMVAR: edit P3 — matched fem_var lookup (NaN when the
            # unit-year has no computable variance; column-level absence
            # was handled above)
            if has_fem_var:
                fv_series = unit_fv[name]
                row["fem_var"] = (
                    fv_series.loc[yr]
                    if (fv_series is not None and yr in fv_series.index)
                    else np.nan)
            panel_rows.append(row)

    base_cols = ["profession", "year", "women_prop", "women_n", "total_n",
                 "women_n_unw", "total_n_unw", "projection"]
    panel_df = pd.DataFrame(
        panel_rows,
        columns=base_cols + (["fem_var"] if has_fem_var else []))
    return panel_df