# utils/display.py
"""Common display formatting utilities for the ngram pipeline."""

from pathlib import Path
from typing import Union

__all__ = ["format_bytes", "truncate_path_to_fit", "format_banner"]


def format_bytes(num_bytes: int) -> str:
    """Convert bytes to human-readable format.

    Args:
        num_bytes: Number of bytes to format.

    Returns:
        Human-readable string with appropriate unit (B, KB, MB, GB, TB, PB).

    Examples:
        >>> format_bytes(1024)
        '1.00 KB'
        >>> format_bytes(1536000)
        '1.46 MB'
        >>> format_bytes(5368709120)
        '5.00 GB'
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def truncate_path_to_fit(
    path: Union[Path, str],
    prefix: str,
    total_width: int = 100,
) -> str:
    """Truncate path to fit within total_width including prefix.

    The total line length (prefix + path) will not exceed total_width.
    Longer prefixes automatically get less space for the path.

    Args:
        path: Path to display
        prefix: The label/prefix before the path
        total_width: Total character width for the entire line

    Returns:
        Truncated path that fits within (total_width - len(prefix))

    Examples:
        >>> truncate_path_to_fit("/long/path", "Short: ", 50)
        '/long/path'  # Fits within 50 - 7 = 43 chars
        >>> truncate_path_to_fit("/very/long/path/to/file.db", "Very long prefix: ", 50)
        '...to/file.db'  # Truncated to fit within 50 - 18 = 32 chars
    """
    path_str = str(path)
    max_path_length = total_width - len(prefix)

    if len(path_str) <= max_path_length:
        return path_str

    # Need at least 4 chars for "..."
    if max_path_length < 4:
        return "..."

    return "..." + path_str[-(max_path_length - 3) :]


def format_banner(title: str, width: int = 100, style: str = "═") -> str:
    """Create a formatted banner with title and separator line.

    Args:
        title: Banner title text
        width: Total width of the banner
        style: Character to use for separator line (═, ━, —, -, etc.)

    Returns:
        Formatted banner string with title and separator

    Examples:
        >>> print(format_banner("Pipeline Start"))
        Pipeline Start
        ════════════════════════════════════════════════════════════════════════
        >>> print(format_banner("Phase 1", width=50, style="─"))
        Phase 1
        ──────────────────────────────────────────────────
    """
    return f"{title}\n{style * width}"


def format_section_header(title: str, width: int = 100, style: str = "—") -> str:
    """Create a formatted section header (lighter style than banner).

    Args:
        title: Section title text
        width: Total width of the header
        style: Character to use for separator line

    Returns:
        Formatted section header

    Examples:
        >>> print(format_section_header("Configuration"))
        Configuration
        ————————————————————————————————————————————————————————————————————————
    """
    return format_banner(title, width, style)
