# tests/pipeline/test_logging_setup.py
from pathlib import Path
import logging
from logging import StreamHandler, FileHandler
from logging.handlers import RotatingFileHandler

import pytest

from ngram_prep.pipeline.logger import setup_logger

pytestmark = pytest.mark.usefixtures("clean_root_handlers")


@pytest.fixture()
def clean_root_handlers():
    """Start each test with a clean root logger; restore afterwards."""
    root = logging.getLogger()
    prev = list(root.handlers)
    try:
        for h in list(root.handlers):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        yield
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        for h in prev:
            root.addHandler(h)


def _handler_types():
    return {type(h) for h in logging.getLogger().handlers}


def test_creates_log_file_and_writes(tmp_path: Path):
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    log_path = setup_logger(db_dir, console=False, force=True)
    assert log_path.parent == db_dir
    assert log_path.suffix == ".log"
    assert log_path.exists()

    logging.getLogger().info("hello world")
    text = log_path.read_text(encoding="utf-8")
    assert "Logging to:" in text
    assert "hello world" in text


def test_uses_parent_dir_when_db_path_is_file(tmp_path: Path):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    db_file = db_dir / "rocks.db"

    log_path = setup_logger(db_file, console=False, force=True)
    assert log_path.parent == db_dir
    assert log_path.exists()


def test_force_replaces_handlers_and_rotation(tmp_path: Path):
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    setup_logger(db_dir, console=True, force=True)
    types1 = _handler_types()
    assert FileHandler in types1 or RotatingFileHandler in types1
    assert StreamHandler in types1

    log_path = setup_logger(db_dir, rotate=True, force=True)
    types2 = _handler_types()
    assert RotatingFileHandler in types2
    assert StreamHandler not in types2

    logging.getLogger().warning("rotate test")
    assert log_path.exists()
    assert "rotate test" in log_path.read_text(encoding="utf-8")
