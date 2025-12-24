"""Display formatting for Word2Vec training pipeline."""

from ngramprep.utilities.display import truncate_path_to_fit

__all__ = [
    "print_training_header",
    "print_completion_banner",
    "print_alignment_header",
    "print_alignment_completion",
    "LINE_WIDTH"
]

LINE_WIDTH = 100


def print_training_header(
        start_time,
        db_path,
        model_dir,
        log_dir,
        max_parallel_models,
        grid_params
):
    """
    Print training configuration header.

    Args:
        start_time (datetime): Start time of the process.
        db_path (str): Database path.
        model_dir (str): Model directory path.
        log_dir (str): Log directory path.
        max_parallel_models (int): Number of parallel models.
        grid_params (str): Formatted grid search parameters.
    """
    # Format paths using truncate_path_to_fit
    db_path_str = truncate_path_to_fit(db_path, "Database:             ", LINE_WIDTH)
    model_dir_str = truncate_path_to_fit(model_dir, "Model directory:      ", LINE_WIDTH)
    log_dir_str = truncate_path_to_fit(log_dir, "Log directory:        ", LINE_WIDTH)

    lines = [
        "WORD2VEC MODEL TRAINING",
        "━" * LINE_WIDTH,
        f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}",
        "",
        "Configuration",
        "═" * LINE_WIDTH,
        f"Database:             {db_path_str}",
        f"Model directory:      {model_dir_str}",
        f"Log directory:        {log_dir_str}",
        f"Parallel models:      {max_parallel_models}",
        "",
        grid_params,
        "",
    ]
    print("\n".join(lines), flush=True)


def print_completion_banner(model_dir, total_tasks):
    """
    Print completion banner with statistics.

    Args:
        model_dir (str): Model directory path.
        total_tasks (int): Total number of models trained.
    """
    model_dir_str = truncate_path_to_fit(model_dir, "Model directory:      ", LINE_WIDTH)

    lines = [
        "",
        "Training Complete",
        "═" * LINE_WIDTH,
        f"Models trained:       {total_tasks}",
        f"Model directory:      {model_dir_str}",
        "━" * LINE_WIDTH,
        "",
    ]
    print("\n".join(lines), flush=True)

def print_alignment_header(
        start_time,
        model_dir,
        output_dir,
        anchor_year,
        num_models,
        weighted_alignment,
        stability_method=None,
        include_frequency=None,
        frequency_weight=None,
        workers=None
):
    """
    Print alignment configuration header.

    Args:
        start_time (datetime): Start time of the process.
        model_dir (str): Input model directory path.
        output_dir (str): Output directory path.
        anchor_year (int): Anchor year for alignment.
        num_models (int): Total number of models to process.
        weighted_alignment (bool): Whether using weighted alignment.
        stability_method (str, optional): Stability computation method.
        include_frequency (bool, optional): Whether including frequency in weights.
        frequency_weight (float, optional): Weight for frequency component.
        workers (int, optional): Number of parallel workers.
    """
    # Format paths using truncate_path_to_fit
    model_dir_str = truncate_path_to_fit(model_dir, "Model directory:      ", LINE_WIDTH)
    output_dir_str = truncate_path_to_fit(output_dir, "Output directory:     ", LINE_WIDTH)

    lines = [
        "",
        "WORD2VEC MODEL NORMALIZATION & ALIGNMENT",
        "━" * LINE_WIDTH,
        f"Start Time: {start_time:%Y-%m-%d %H:%M:%S}",
        "",
        "Configuration",
        "═" * LINE_WIDTH,
        f"Model directory:      {model_dir_str}",
        f"Output directory:     {output_dir_str}",
        f"Anchor year:          {anchor_year}",
        f"Total models:         {num_models}",
    ]

    if workers is not None:
        lines.append(f"Parallel workers:     {workers}")

    lines.append("")
    lines.append("Alignment Method")
    lines.append("─" * LINE_WIDTH)

    if weighted_alignment:
        lines.append(f"Type:                 Weighted Procrustes")
        if stability_method:
            lines.append(f"Stability metric:     {stability_method}")
        if include_frequency is not None:
            freq_status = "Yes" if include_frequency else "No"
            lines.append(f"Include frequency:    {freq_status}")
        if include_frequency and frequency_weight is not None:
            lines.append(f"Frequency weight:     {frequency_weight:.2f} ({int(frequency_weight*100)}% frequency, {int((1-frequency_weight)*100)}% stability)")
    else:
        lines.append(f"Type:                 Unweighted Procrustes (all shared vocabulary)")

    lines.append("")
    print("\n".join(lines), flush=True)


def print_alignment_completion(output_dir, num_models, runtime):
    """
    Print alignment completion banner with statistics.

    Args:
        output_dir (str): Output directory path.
        num_models (int): Total number of models processed.
        runtime (timedelta): Total runtime.
    """
    output_dir_str = truncate_path_to_fit(output_dir, "Output directory:     ", LINE_WIDTH)

    lines = [
        "",
        "Alignment Complete",
        "═" * LINE_WIDTH,
        f"Models processed:     {num_models}",
        f"Output directory:     {output_dir_str}",
        f"Total runtime:        {runtime}",
        "━" * LINE_WIDTH,
        "",
    ]
    print("\n".join(lines), flush=True)
