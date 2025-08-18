def safe_db_cleanup(
    db_path: Union[str, "Path"],
    max_retries: int = 5,
) -> bool:
    """
    Remove a RocksDB directory safely, handling lingering NFS temp files.

    Parameters
    ----------
    db_path : str | Path
        Path to the database directory.
    max_retries : int, default=5
        Number of attempts to remove the directory.

    Returns
    -------
    bool
        True if the directory is removed or doesn't exist; False otherwise.
    """
    path = Path(db_path)

    if not path.exists():
        return True

    delay_seconds = 2.0

    for attempt in range(1, max_retries + 1):
        try:
            for nfs_file in path.glob(".nfs*"):
                with suppress(OSError):
                    nfs_file.unlink()
                    logger.info("Removed NFS temp file: %s", nfs_file.name)

            import shutil
            shutil.rmtree(path)

            logger.info("Successfully removed database (attempt %d)", attempt)
            return True

        except OSError as exc:
            if attempt < max_retries:
                logger.warning(
                    "Database cleanup attempt %d/%d failed: %s",
                    attempt,
                    max_retries,
                    exc,
                )
                logger.info("Retrying in %.1f seconds...", delay_seconds)
                time.sleep(delay_seconds)
            else:
                logger.error(
                    "Failed to remove database after %d attempts: %s",
                    max_retries,
                    exc,
                )
                return False

    return False