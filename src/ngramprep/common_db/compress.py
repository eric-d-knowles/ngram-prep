"""Compress RocksDB directories using zstd with parallel processing."""

from __future__ import annotations

import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Optional


def compress_db(
    db_path: Path | str,
    output_path: Optional[Path | str] = None,
    compression_level: int = 3,
    num_threads: Optional[int] = None,
    independent_frames: bool = True,
) -> Path:
    """
    Compress a RocksDB directory using zstd with parallel processing.

    Args:
        db_path: Path to the RocksDB directory to compress
        output_path: Output path for compressed archive. If None, uses db_path.tar.zst
        compression_level: Zstd compression level (1-22, default 3)
        num_threads: Number of threads to use. If None, uses all available CPUs
        independent_frames: If True, creates independent frames for better parallel decompression

    Returns:
        Path to the compressed archive

    Raises:
        FileNotFoundError: If db_path does not exist
        subprocess.CalledProcessError: If compression fails
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database directory not found: {db_path}")

    if not db_path.is_dir():
        raise ValueError(f"Path is not a directory: {db_path}")

    # Determine output path
    if output_path is None:
        output_path = db_path.with_suffix('.tar.zst')
    else:
        output_path = Path(output_path)

    # Determine number of threads
    if num_threads is None:
        num_threads = mp.cpu_count()

    # Build tar command (create archive and pipe to zstd)
    tar_cmd = ["tar", "-cf", "-", "-C", str(db_path.parent), db_path.name]

    # Build zstd command
    zstd_cmd = [
        "zstd",
        f"-{compression_level}",
        f"-T{num_threads}",
        "-o", str(output_path),
    ]

    # Add independent frames flag for parallel decompression
    if independent_frames:
        zstd_cmd.insert(2, "--rsyncable")

    # Create tar archive and pipe to zstd
    print(f"Compressing {db_path} to {output_path}")
    print(f"Using {num_threads} threads at compression level {compression_level}")

    tar_proc = subprocess.Popen(tar_cmd, stdout=subprocess.PIPE)
    zstd_proc = subprocess.Popen(
        zstd_cmd,
        stdin=tar_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Close tar stdout in parent to allow tar to receive SIGPIPE if zstd exits
    if tar_proc.stdout:
        tar_proc.stdout.close()

    # Wait for completion
    _, stderr = zstd_proc.communicate()
    tar_returncode = tar_proc.wait()

    # Check for errors
    if tar_returncode != 0:
        raise subprocess.CalledProcessError(tar_returncode, tar_cmd)

    if zstd_proc.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        raise subprocess.CalledProcessError(
            zstd_proc.returncode,
            zstd_cmd,
            stderr=error_msg,
        )

    print(f"Compression complete: {output_path}")
    return output_path


def decompress_db(
    archive_path: Path | str,
    output_dir: Optional[Path | str] = None,
    num_threads: Optional[int] = None,
) -> Path:
    """
    Decompress a zstd-compressed RocksDB archive.

    Args:
        archive_path: Path to the .tar.zst archive
        output_dir: Directory to extract to. If None, extracts to current directory
        num_threads: Number of threads to use. If None, uses all available CPUs

    Returns:
        Path to the extracted directory

    Raises:
        FileNotFoundError: If archive_path does not exist
        subprocess.CalledProcessError: If decompression fails
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of threads
    if num_threads is None:
        num_threads = mp.cpu_count()

    # Build zstd decompress command
    zstd_cmd = [
        "zstd",
        "-d",
        f"-T{num_threads}",
        "-c",
        str(archive_path),
    ]

    # Build tar extract command
    tar_cmd = ["tar", "-xf", "-", "-C", str(output_dir)]

    print(f"Decompressing {archive_path} to {output_dir}")
    print(f"Using {num_threads} threads")

    # Decompress with zstd and pipe to tar
    zstd_proc = subprocess.Popen(zstd_cmd, stdout=subprocess.PIPE)
    tar_proc = subprocess.Popen(
        tar_cmd,
        stdin=zstd_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Close zstd stdout in parent
    if zstd_proc.stdout:
        zstd_proc.stdout.close()

    # Wait for completion
    _, stderr = tar_proc.communicate()
    zstd_returncode = zstd_proc.wait()

    # Check for errors
    if zstd_returncode != 0:
        raise subprocess.CalledProcessError(zstd_returncode, zstd_cmd)

    if tar_proc.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        raise subprocess.CalledProcessError(
            tar_proc.returncode,
            tar_cmd,
            stderr=error_msg,
        )

    # Determine extracted path (assume it's the archive name without extensions)
    extracted_name = archive_path.name.replace('.tar.zst', '').replace('.tar', '')
    extracted_path = output_dir / extracted_name

    print(f"Decompression complete: {extracted_path}")
    return extracted_path
