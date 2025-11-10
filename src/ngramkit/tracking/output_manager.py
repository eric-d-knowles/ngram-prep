"""Output management for worker processes."""

from __future__ import annotations

from pathlib import Path

__all__ = ["SimpleOutputManager"]


class SimpleOutputManager:
    """Simple file-based output manager for database shards or similar outputs."""

    def __init__(self, output_dir: Path, extension: str = ".db"):
        """Initialize output manager.

        Args:
            output_dir: Directory where outputs are stored
            extension: File extension for outputs (e.g., '.db', '.parquet')
        """
        self.output_dir = Path(output_dir)
        self.extension = extension
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_output_path(self, unit_id: str) -> Path:
        """Get output path for a work unit."""
        return self.output_dir / f"{unit_id}{self.extension}"

    def cleanup_partial_output(self, unit_id: str) -> None:
        """Remove partial output if it exists."""
        import shutil

        output_path = self.get_output_path(unit_id)
        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()

    def output_exists(self, unit_id: str) -> bool:
        """Check if output exists."""
        return self.get_output_path(unit_id).exists()
