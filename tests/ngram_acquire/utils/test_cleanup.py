# tests/test_cleanup.py
from pathlib import Path
import os
import shutil
import logging

import pytest

# Adjust this import if your module path differs
from ngram_acquire.utils.cleanup import safe_db_cleanup


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Disable real sleeping to keep tests fast."""
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda *_: None)


def test_returns_true_when_path_missing(tmp_path: Path):
    missing = tmp_path / "nope"
    assert safe_db_cleanup(missing) is True


def test_raises_if_target_is_not_directory(tmp_path: Path):
    file_path = tmp_path / "file"
    file_path.write_text("x")
    with pytest.raises(ValueError):
        safe_db_cleanup(file_path)


def test_removes_directory_once_and_returns_true(tmp_path: Path, monkeypatch):
    target = tmp_path / "dbdir"
    (target / "sub").mkdir(parents=True)
    (target / "sub" / "file.txt").write_text("x")

    calls = {"rmtree": 0}

    def fake_rmtree(p):
        assert Path(p) == target
        calls["rmtree"] += 1
        # simulate successful removal
        for root, dirs, files in os.walk(target, topdown=False):
            for name in files:
                (Path(root) / name).unlink()
            for name in dirs:
                (Path(root) / name).rmdir()
        target.rmdir()

    monkeypatch.setattr(shutil, "rmtree", fake_rmtree)

    assert safe_db_cleanup(target) is True
    assert calls["rmtree"] == 1
    assert not target.exists()


def test_removes_nfs_temp_files_when_unlink_succeeds(tmp_path: Path, monkeypatch):
    target = tmp_path / "dbdir"
    target.mkdir()
    (target / ".nfs123").write_text("a")
    (target / ".nfs999").write_text("b")

    removed = []
    orig_unlink = Path.unlink

    def tracking_unlink(self, *args, **kwargs):
        removed.append(self.name)
        return orig_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", tracking_unlink, raising=True)
    monkeypatch.setattr(shutil, "rmtree", lambda p: target.rmdir())

    assert safe_db_cleanup(target) is True
    assert {".nfs123", ".nfs999"} <= set(removed)
    assert not target.exists()


def test_busy_nfs_files_are_renamed_then_cleanup_succeeds(
    tmp_path: Path, monkeypatch, caplog
):
    """
    Simulate NFS: unlink of .nfs* fails -> function should rename into parent,
    allowing directory removal to succeed.
    """
    caplog.set_level(logging.DEBUG)

    target = tmp_path / "dbdir"
    target.mkdir()
    (target / ".nfs_keep").write_text("x")

    def failing_unlink(self, *args, **kwargs):
        if self.name.startswith(".nfs"):
            raise OSError("busy file")
        return Path.unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink, raising=True)
    monkeypatch.setattr(shutil, "rmtree", lambda p: target.rmdir())

    assert safe_db_cleanup(target) is True
    assert not target.exists()
    # Verify we observed a rename path (if implementation logs it)
    assert any("moved to" in rec.getMessage().lower() for rec in caplog.records)


def test_retries_then_succeeds(tmp_path: Path, monkeypatch, caplog):
    target = tmp_path / "dbdir"
    target.mkdir()

    calls = {"rmtree": 0}

    def flaky_rmtree(p):
        calls["rmtree"] += 1
        if calls["rmtree"] == 1:
            raise OSError("first try fails")
        target.rmdir()

    monkeypatch.setattr(shutil, "rmtree", flaky_rmtree)

    with caplog.at_level("WARNING"):
        assert safe_db_cleanup(target, max_retries=3) is True

    assert calls["rmtree"] == 2
    assert any(
        "cleanup attempt 1/3 failed" in rec.getMessage().lower()
        for rec in caplog.records
    )


def test_exhausts_retries_and_returns_false(tmp_path: Path, monkeypatch):
    target = tmp_path / "dbdir"
    target.mkdir()

    def always_fail(_):
        raise OSError("nope")

    monkeypatch.setattr(shutil, "rmtree", always_fail)

    sleeps = []
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: sleeps.append(s))

    assert safe_db_cleanup(target, max_retries=3) is False
    # slept between 1→2 and 2→3
    assert len(sleeps) == 2
    assert target.exists()
