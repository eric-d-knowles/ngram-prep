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

    def _handle_mode(self) -> None:
        """Handle different execution modes (restart/resume/reprocess)."""
        mode = self.pipeline_config.mode
        tmp_dir = self.temp_paths['tmp_dir']
        dst_db = self.temp_paths['dst_db']

        if mode == "restart":
            # Clean restart: delete everything and start fresh
            if dst_db.exists():
                shutil.rmtree(dst_db, ignore_errors=True)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            self.temp_paths['output_dir'].mkdir(parents=True, exist_ok=True)

        elif mode == "reprocess":
            # Reprocess: keep work tracker but reset status
            if dst_db.exists():
                shutil.rmtree(dst_db, ignore_errors=True)
            # Keep tmp_dir for work tracker
            tmp_dir.mkdir(parents=True, exist_ok=True)
            self.temp_paths['output_dir'].mkdir(parents=True, exist_ok=True)

        elif mode == "resume":
            # Resume: keep everything, just create if missing
            tmp_dir.mkdir(parents=True, exist_ok=True)
            self.temp_paths['output_dir'].mkdir(parents=True, exist_ok=True)

        else:
            raise ValueError(f"Invalid mode: {mode}")
