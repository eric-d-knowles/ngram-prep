# tests/ngram_filter/pipeline/test_writers.py
from __future__ import annotations

import multiprocessing as mp
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

import ngram_filter.pipeline.old.writers as mod
from ngram_filter.pipeline.old.routing import Router


# ---------- mp "fork" so monkeypatches propagate into children ----------
@pytest.fixture(autouse=True)
def mp_fork():
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    yield


# ---------- Fakes / helpers ----------
@dataclass
class FakeDB:
    store: Dict[bytes, bytes]

    def write_batch(self, *_, **__):
        return FakeBatch(self)


class FakeBatch:
    def __init__(self, db: FakeDB):
        self.db = db
        self.ops: List[Tuple[str, bytes, bytes]] = []  # list of (op, k, v)

    def merge(self, k: bytes, v: bytes):
        self.ops.append(("merge", k, v or b""))

    def put(self, k: bytes, v: bytes):
        self.ops.append(("put", k, v or b""))

    def commit(self):
        for op, k, v in self.ops:
            if op == "merge":
                self.db.store[k] = self.db.store.get(k, b"") + v
            else:
                self.db.store[k] = v
        self.ops.clear()


class _NullLock:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


class Box:
    """mp.Value-like test stub with .value and .get_lock()."""
    def __init__(self, v: int = 0):
        self.value = v
        self._lock = _NullLock()
    def get_lock(self):
        return self._lock


@dataclass
class DummyCounters:
    # Match the fields writers uses
    items_written: Box = field(default_factory=Box)
    bytes_flushed: Box = field(default_factory=Box)
    items_buffered: Box = field(default_factory=Box)
    opened_dbs: Box = field(default_factory=Box)
    closed_dbs: Box = field(default_factory=Box)


@pytest.fixture
def counters():
    # Box()-backed counters so mod._inc(...) works unchanged
    return DummyCounters()


@pytest.fixture
def fake_db_env(monkeypatch, tmp_path):
    """
    Patch shim-backed open_db to yield a FakeDB and capture created DBs.
    """
    created: List[FakeDB] = []

    @contextmanager
    def _open_db(db_path: Path, *_, **__):
        db = FakeDB(store={})
        created.append(db)
        yield db

    monkeypatch.setattr(mod, "open_db", _open_db)

    return {"created": created, "root": tmp_path}


# ---------- Unit tests for _flush_lane_to_db ----------
def test_flush_lane_writes_all_ops_and_updates_counters(counters):
    # Prepare a lane with two items
    db = FakeDB(store={})
    lane = [(b"a", b"1", "merge"), (b"b", b"22", "merge")]

    # Call the helper directly
    bytes_written = mod._flush_lane_to_db(db, lane, counters)

    # Bytes = sum(len(k)+len(v))
    assert bytes_written == (len(b"a") + len(b"1") + len(b"b") + len(b"22"))
    # Lane cleared
    assert lane == []
    # Store contains both records (MERGE w/ empty existing -> just puts)
    assert db.store == {b"a": b"1", b"b": b"22"}
    # Counters reflect number of ops written, not unique keys
    assert counters.items_written.value == 2
    assert counters.bytes_flushed.value == bytes_written


# ---------- Integration-ish tests for writer_worker ----------
def test_writer_worker_merges_duplicates_in_db_and_flushes_at_end(monkeypatch, counters, fake_db_env):
    q = mp.Queue()
    router = Router(num_shards=1, inner_lanes=1, seed=0)

    # same key twice; MERGE concatenation -> expect b"ab" in store
    q.put((b"k", b"a"))
    q.put((b"k", b"b"))
    q.put(None)

    mod.writer_worker(
        shard_id=0,
        q=q,
        db_root=fake_db_env["root"],
        router=router,
        cfg=mod.WriterConfig(max_items_per_lane=0, max_bytes_per_lane=0),  # only final flush
        counters=counters,
    )

    db = fake_db_env["created"][0]
    assert db.store == {b"k": b"ab"}          # MERGE semantics applied
    assert counters.opened_dbs.value == 1
    assert counters.closed_dbs.value == 1
    assert counters.items_buffered.value == 2  # two queue items seen
    assert counters.items_written.value == 2   # two ops written (no RAM combining)


def test_writer_worker_flushes_on_item_threshold(monkeypatch, counters, fake_db_env):
    q = mp.Queue()
    router = Router(num_shards=1, inner_lanes=1, seed=0)

    # With max_items_per_lane=1, each append triggers a flush
    q.put((b"a", b"1"))
    q.put((b"b", b"2"))
    q.put(None)

    mod.writer_worker(
        shard_id=0,
        q=q,
        db_root=fake_db_env["root"],
        router=router,
        cfg=mod.WriterConfig(max_items_per_lane=1, max_bytes_per_lane=0),
        counters=counters,
    )

    db = fake_db_env["created"][0]
    assert db.store == {b"a": b"1", b"b": b"2"}
    assert counters.items_buffered.value == 2
    assert counters.items_written.value == 2
    assert counters.opened_dbs.value == 1
    assert counters.closed_dbs.value == 1


def _find_two_keys_in_distinct_lanes(router: Router, max_tries: int = 100000) -> Tuple[bytes, bytes]:
    """
    Deterministically search for two keys that land in different inner lanes.
    Avoids monkeypatching Router.route on a frozen dataclass.
    """
    want = None
    for i in range(max_tries):
        k = f"a{i}".encode()
        lane = router.route(k)[1]
        if want is None:
            want = (k, lane)
        elif lane != want[1]:
            return want[0], k
    raise RuntimeError("Could not find keys in distinct lanes; increase max_tries")


def test_writer_worker_uses_multiple_inner_lanes(monkeypatch, counters, fake_db_env):
    q = mp.Queue()
    router = Router(num_shards=1, inner_lanes=2, seed=12345)

    k0, k1 = _find_two_keys_in_distinct_lanes(router)
    q.put((k0, b"A"))
    q.put((k1, b"B"))
    q.put(None)

    mod.writer_worker(
        shard_id=3,
        q=q,
        db_root=fake_db_env["root"],
        router=router,
        cfg=mod.WriterConfig(max_items_per_lane=0, max_bytes_per_lane=0),
        counters=counters,
    )

    db = fake_db_env["created"][0]
    assert db.store == {k0: b"A", k1: b"B"}
    assert counters.items_buffered.value == 2
    assert counters.items_written.value == 2


def test_writer_worker_flushes_on_byte_threshold(monkeypatch, counters, fake_db_env):
    q = mp.Queue()
    router = Router(num_shards=1, inner_lanes=1, seed=0)

    # Each insert ~ len(k)+len(v) bytes. With max_bytes_per_lane=2, the first insert
    # will meet/exceed threshold and flush, the second will be buffered and flushed at end.
    q.put((b"a", b"1"))  # 2 bytes total
    q.put((b"b", b"2"))
    q.put(None)

    mod.writer_worker(
        shard_id=0,
        q=q,
        db_root=fake_db_env["root"],
        router=router,
        cfg=mod.WriterConfig(max_items_per_lane=0, max_bytes_per_lane=2),
        counters=counters,
    )

    db = fake_db_env["created"][0]
    assert db.store == {b"a": b"1", b"b": b"2"}
    assert counters.items_written.value == 2
    assert counters.opened_dbs.value == 1
    assert counters.closed_dbs.value == 1


def test_writer_entry_returns_summary_via_pipe(monkeypatch, counters, fake_db_env):
    router = Router(num_shards=1, inner_lanes=1, seed=0)
    q = mp.Queue()
    q.put((b"k", b"v"))
    q.put(None)

    parent, child = mp.Pipe(duplex=False)

    p = mp.Process(
        target=mod.writer_entry,
        kwargs=dict(
            shard_id=5,
            q=q,
            db_root=fake_db_env["root"],
            router=router,
            conn=child,
            cfg=mod.WriterConfig(),
            counters=counters,
        ),
    )
    p.start()
    assert parent.poll(5), "Child process did not send summary within 5s"
    summary = parent.recv()
    p.join()

    assert summary == {"shard_id": 5, "ok": True}
