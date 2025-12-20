"""Professional logging setup for ngram-kit notebooks."""

from pathlib import Path
from datetime import datetime


def setup_notebook_logging(
    workflow_name,
    data_path=None,
    log_base_dir="/scratch/edk202/ngram-kit/logs",
    console=True,
    rotate=True,
    max_bytes=100_000_000,
    backup_count=5
):
    """
    Set up logging for a notebook with consistent naming and location.

    Args:
        workflow_name: Name of the workflow (e.g., "download_5grams", "process_unigrams")
        data_path: Optional path to data being processed (for context in logs)
        log_base_dir: Base directory for all logs (default: /scratch/edk202/ngram-kit/logs)
        console: Whether to also log to console
        rotate: Whether to rotate log files
        max_bytes: Max size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Path to the log file
    """
    from ngramprep.ngram_acquire.logger import setup_logger
    import logging

    # Create log directory structure
    log_dir = Path(log_base_dir) / workflow_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    log_file = setup_logger(
        db_path=str(log_dir),
        filename_prefix=workflow_name,
        console=console,
        rotate=rotate,
        max_bytes=max_bytes,
        backup_count=backup_count,
        force=True
    )

    # Log initial context
    logger = logging.getLogger()
    logger.info(f"=" * 80)
    logger.info(f"Workflow: {workflow_name}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if data_path:
        logger.info(f"Data path: {data_path}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"=" * 80)

    return log_file
