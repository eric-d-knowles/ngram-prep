import os
from pathlib import Path

# Adjust this import if your module path differs
from ngram_prep.utils.cleanup import safe_db_cleanup


def test_returns_true_when_path_does_not_exist(tmp_path: Path):
    missing = tmp_path / "nope"
    assert not missing.exists()
    assert safe_db_cleanup(missing) is True


def test_removes_directory_and_returns_true(tmp_path: Path, monkeypatch):
    target = tmp_path / "dbdir"
    (target / "sub").mkdir(parents=True)
    (target / "sub" / "file.txt").write_text("x")

    calls = {"rmtree": 0}

    def fake_rmtree(p):
        assert Path(p) == target
        calls["rmtree"] += 1
        # simulate successful removal by actually removing
        for root, dirs, files in os.walk(target, topdown=False):
            for name in files:
                (Path(root) / name).unlink()
            for name in dirs:
                (Path(root) / name).rmdir()
        target.rmdir()

    import shutil

    monkeypatch.setattr(shutil, "rmtree", fake_rmtree)

    assert safe_db_cleanup(str(target)) is True
    assert calls["rmtree"] == 1
    assert not target.exists()


def test_removes_nfs_temp_files(tmp_path: Path, monkeypatch):
    target = tmp_path / "dbdir"
    target.mkdir()
    nfs1 = target / ".nfs123"
    nfs2 = target / ".nfs999"
    nfs1.write_text("a")
    nfs2.write_text("b")

    removed = []

    def fake_unlink(self):
        removed.append(self.name)
        Path(self).unlink(missing_ok=False)

    monkeypatch.setattr(Path, "unlink", fake_unlink, raising=True)

    # also avoid actually deleting the dir; pretend success
    import shutil

    monkeypatch.setattr(shutil, "rmtree", lambda p: target.rmdir())

    assert safe_db_cleanup(target) is True
    assert set(removed) >= {".nfs123", ".nfs999"}
    assert not target.exists()


def test_unlink_errors_are_suppressed(tmp_path: Path, monkeypatch):
    target = tmp_path / "dbdir"
    target.mkdir()
    (target / ".nfs_keep").write_text("x")

    # Make Path.unlink raise for .nfs files
    def failing_unlink(self):
        raise OSError("busy file")

    monkeypatch.setattr(Path, "unlink", failing_unlink, raising=True)

    import shutil

    monkeypatch.setattr(shutil, "rmtree", lambda p: target.rmdir())

    # Should still succeed despite unlink errors (suppressed)
    assert safe_db_cleanup(target) is True
    assert not target.exists()


def test_retries_then_succeeds(tmp_path: Path, monkeypatch, caplog):
    target = tmp_path / "dbdir"
    target.mkdir()

    import shutil

    calls = {"rmtree": 0}

    def flaky_rmtree(p):
        calls["rmtree"] += 1
        if calls["rmtree"] == 1:
            raise OSError("first try fails")
        target.rmdir()

    monkeypatch.setattr(shutil, "rmtree", flaky_rmtree)

    # no real sleeping
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *_: None)

    with caplog.at_level("WARNING"):
        assert safe_db_cleanup(target, max_retries=3) is True

    assert calls["rmtree"] == 2
    assert any("cleanup attempt 1/3 failed" in msg.lower() for _, _, msg in caplog.record_tuples)


def test_exhausts_retries_and_returns_false(tmp_path: Path, monkeypatch):
    target = tmp_path / "dbdir"
    target.mkdir()

    import shutil

    monkeypatch.setattr(shutil, "rmtree", lambda p: (_ for _ in ()).throw(OSError("nope")))

    # stub sleep so tests are fast
    import time as _time

    sleeps = []
    monkeypatch.setattr(_time, "sleep", lambda s: sleeps.append(s))

    assert safe_db_cleanup(target, max_retries=3) is False
    # should have slept between first and second, second and third attempt
    assert len(sleeps) == 2
    assert target.exists()  # still there since we always failed
