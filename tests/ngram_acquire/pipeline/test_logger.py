# tests/pipeline/test_logger.py
import logging
from pathlib import Path
from ngram_acquire.pipeline.logger import setup_logger


def test_creates_log_file(tmp_path):
    dbdir = tmp_path / "dbdir"
    dbdir.mkdir()
    log_path = setup_logger(dbdir, force=True)
    assert log_path.exists()
    assert log_path.read_text().strip() != ""


def test_force_removes_existing_handlers(tmp_path):
    dbdir = tmp_path / "dbdir"
    dbdir.mkdir()
    # first call adds a handler
    path1 = setup_logger(dbdir, force=True)
    root = logging.getLogger()
    assert any(isinstance(h, logging.FileHandler) for h in root.handlers)
    n = len(root.handlers)
    # second call with force should reset handlers
    path2 = setup_logger(dbdir, force=True)
    assert len(root.handlers) <= n


def test_console_and_rotate(tmp_path):
    dbdir = tmp_path / "dbdir"
    dbdir.mkdir()
    log_path = setup_logger(dbdir, console=True, rotate=True, force=True)
    assert log_path.exists()
    root = logging.getLogger()
    # check both console and file handlers are present
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    assert any(isinstance(h, logging.FileHandler) for h in root.handlers)


def test_file_location_from_file_path(tmp_path):
    dbfile = tmp_path / "data.db"
    dbfile.write_text("dummy")
    log_path = setup_logger(dbfile, force=True)
    # should have chosen parent directory
    assert log_path.parent == tmp_path
