"""Setup and resource preparation for the pivot pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

from ..config import PipelineConfig

__all__ = ["PipelineSetup"]


class PipelineSetup:
    """Handles initialization and preparation of pipeline resources."""

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize pipeline setup.

        Args:
            pipeline_config: Configuration for pipeline execution
        """
        self.pipeline_config = pipeline_config
        self.temp_paths: Dict[str, Path] = {}

    def prepare_all(self) -> Dict[str, Path]:
        """
        Prepare all resources needed for pipeline execution.

        Returns:
            Dictionary mapping resource names to paths
        """
        self._setup_paths()
        self._handle_mode()
        return self.temp_paths

    def _setup_paths(self) -> None:
        """Set up all paths used by the pipeline."""
        tmp_dir = self.pipeline_config.tmp_dir

        self.temp_paths = {
            'src_db': self.pipeline_config.src_db,
            'dst_db': self.pipeline_config.dst_db,
            'tmp_dir': tmp_dir,
            'work_tracker': tmp_dir / "work_tracker.db",
            'output_dir': tmp_dir / "shards",
        }

        # Store partition cache in parent directory (not in tmp_dir) so it survives restarts
        # The cache is based on source DB characteristics, not processing state
        self.temp_paths['cache_dir'] = tmp_dir.parent / ".partition_cache"
        self.temp_paths['base'] = self.temp_paths['cache_dir']  # Used by PartitionCache

    def _handle_mode(self) -> None:
        """Handle different execution modes (restart/resume/reprocess).

        Mode behavior:
            - "restart": Wipes output DB and temp directory (but NOT cache directory)
            - "resume": Preserves all existing state
            - "reprocess": Wipes output DB but preserves temp directory (and cache)
        """
        mode = self.pipeline_config.mode
        tmp_dir = self.temp_paths['tmp_dir']
        dst_db = self.temp_paths['dst_db']

        # Clean destination DB for restart/reprocess modes
        if mode in ("restart", "reprocess"):
            if dst_db.exists():
                shutil.rmtree(dst_db, ignore_errors=True)

        # Ensure parent directory exists for dst_db
        dst_db.parent.mkdir(parents=True, exist_ok=True)

        # Clean temp directory only for restart mode (reprocess preserves cache and work tracker)
        if mode == "restart":
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

        # Create temp directories
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_paths['output_dir'].mkdir(parents=True, exist_ok=True)

        # Create cache directory (survives restarts, outside of tmp_dir)
        self.temp_paths['cache_dir'].mkdir(parents=True, exist_ok=True)
