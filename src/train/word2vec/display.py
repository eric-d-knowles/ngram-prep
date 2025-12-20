"""Display formatting for Word2Vec training pipeline."""

from ngramprep.utilities.display import truncate_path_to_fit

__all__ = ["print_training_header", "print_completion_banner", "LINE_WIDTH"]

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
        "",
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
