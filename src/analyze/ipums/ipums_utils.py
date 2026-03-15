def _aggregate_one_year(args):
    """
    Worker function: aggregate a single year-slice DataFrame into a BLS CSV.
    Receives a pre-sliced DataFrame rather than re-reading the extract file.
    """
    year, year_df, output_csv, kwargs = args
    import tempfile, os

    # Write the year-slice to a temporary parquet file so aggregate_ipums_professions_csv
    # can read it via the existing _read_ipums_extract path without modification.
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        year_df.to_parquet(tmp_path, index=False)
        out_df, out_year = aggregate_ipums_professions_csv(
            extract_file=tmp_path,
            output_csv=output_csv,
            year=None,  # already filtered — don't filter again
            return_year=True,
            **kwargs,
        )
        return {
            "year":       out_year,
            "rows":       len(out_df),
            "status":     "ok",
            "output_csv": out_df.attrs.get("output_csv", ""),
            "error":      "",
        }
    except Exception as exc:
        return {
            "year":       year,
            "rows":       None,
            "status":     "failed",
            "output_csv": "",
            "error":      str(exc),
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def aggregate_ipums_professions_csv_batch(
    extract_file,
    output_dir,
    years=None,
    output_basename="professionsIPUMS.csv",
    continue_on_error=True,
    num_workers=None,
    **kwargs,
):
    """Export one BLS-compatible CSV per year from an IPUMS extract."""
    import multiprocessing as mp

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Read the extract once and split by year in memory — avoids re-reading
    # the full file on every iteration of the original sequential loop.
    df = _read_ipums_extract(extract_file)
    year_col = kwargs.get("year_col", "YEAR")

    if years is None:
        if year_col not in df.columns:
            raise ValueError("years=None requires a year column in extract")
        year_values = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
        years = sorted(year_values.unique().tolist())

    if year_col in df.columns:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        year_slices = {
            year: df[df[year_col] == year].copy()
            for year in years
        }
    else:
        # No year column — every year gets the full DataFrame (single-year extract)
        year_slices = {year: df.copy() for year in years}

    output_csv = str(output_dir_path / output_basename)

    tasks = [
        (year, year_slices.get(year, pd.DataFrame()), output_csv, kwargs)
        for year in years
    ]

    if num_workers is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        num_workers = int(slurm_cpus) if slurm_cpus else min(mp.cpu_count(), len(tasks))

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        results = pool.map(_aggregate_one_year, tasks)

    if not continue_on_error:
        for r in results:
            if r["status"] == "failed":
                raise RuntimeError(
                    f"Aggregation failed for year {r['year']}: {r['error']}"
                )

    runs_df = pd.DataFrame(results)
    if runs_df.empty:
        return runs_df

    return runs_df[["year", "rows", "status", "output_csv", "error"]].sort_values(
        by=["year", "status"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
