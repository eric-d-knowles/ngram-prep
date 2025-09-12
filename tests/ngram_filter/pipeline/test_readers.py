# tests/ngram_filter/pipeline/test_ingest.py
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import ngram_filter.pipeline.ingest as mod

# ---------- Fakes / helpers ----------

@dataclass
class FakeIter:
    items: List[Tuple[bytes, bytes]]
    i: int = 0
    def seek(self, lower: bytes):
        self.i = 0
        for j, (k, _) in enumerate(self.items):
            if k >= lower:
                self.i = j
                break
    def valid(self): return self.i < len(self.items)
    def key(self): return self.items[self.i][0]
    def value(self): return self.items[self.i][1]
    def next(self): self.i += 1

class FakeDB:
    def __init__(self, items: List[Tuple[bytes, bytes]] | None = None):
        self.store: Dict[bytes, bytes] = {}
        self._it = FakeIter(items or [])
    def iterator(self): return self._it
    def write_batch(self, *_, **__): return FakeBatch(self)
    def compact_all(self): pass  # no-op for test

class FakeBatch:
    def __init__(self, db: FakeDB):
        self.db = db
        self.ops: List[Tuple[str, bytes, bytes]] = []
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

def _scan_all(db: FakeDB):
    # mimic common_db.api.scan_all: forward scan of all key/value pairs
    it = db.iterator()
    it.seek(b"")
    out: List[Tuple[bytes, bytes]] = []
    while it.valid():
        out.append((it.key(), it.value()))
        it.next()
    return out

# ---------- Test ----------

def test_ingest_shards_streaming_merges_sources(monkeypatch, tmp_path):
    # Create two shard dirs the orchestrator/ingest expects (directories named shard-*.db)
    shards_root = tmp_path / "shards_stage"
    shards_root.mkdir()
    shard0 = shards_root / "shard-0.db"
    shard1 = shards_root / "shard-1.db"
    shard0.mkdir()
    shard1.mkdir()

    # Prepare source contents (sorted keys). Overlap on key b"a" to exercise MERGE.
    src_contents = {
        shard0: [(b"a", b"1"), (b"b", b"2")],
        shard1: [(b"a", b"3"), (b"c", b"4")],
    }

    created: Dict[Path, FakeDB] = {}
    dst_db_path = tmp_path / "final.db"

    @contextmanager
    def _open_db(path: Path, *_, **__):
        # Destination DB: single instance per dst_db_path
        if path == dst_db_path:
            db = created.setdefault(path, FakeDB())
            yield db
            return
        # Source shard DBs: return DBs seeded with their items
        if path in src_contents:
            yield FakeDB(src_contents[path])
            return
        # Unknown path – create empty
        db = created.setdefault(path, FakeDB())
        yield db

    monkeypatch.setattr(mod, "open_db", _open_db)
    monkeypatch.setattr(mod, "scan_all", _scan_all)

    # Act
    mod.ingest_shards_streaming(dst_db_path, shards_root, batch_bytes=1 << 20, batch_items=10_000)

    # Assert: MERGE concatenation across shards: a -> b"1"+b"3" = b"13"
    dst = created[dst_db_path]
    assert dst.store == {b"a": b"13", b"b": b"2", b"c": b"4"}
