# tests/db/test_write_shim_variants.py
import pytest
from ngram_acquire.db.write import write_batch_to_db

class WB_put:
    def __init__(self): self.ops=[]
    def put(self, k, v): self.ops.append((k, v))

class WB_put_cf:
    def __init__(self): self.ops=[]
    def put_cf(self, cf, k, v): self.ops.append((cf, k, v))

class DB_write_batch_with_put:
    def __init__(self): self.store={}
    def write_batch(self, wb: WB_put):
        for k, v in wb.ops: self.store[k]=v

class DB_write_batch_with_put_cf:
    def __init__(self):
        self.store={}
        self.default_cf = object()
    def write_batch(self, wb: WB_put_cf):
        for cf, k, v in wb.ops:
            assert cf is self.default_cf
            self.store[k]=v

def test_writebatch_works_with_put(monkeypatch):
    import rocks_shim as rs
    monkeypatch.setattr(rs, "WriteBatch", WB_put, raising=True)
    db = DB_write_batch_with_put()
    wrote = write_batch_to_db(db, {"a": b"1", b"b": b"2"})
    assert wrote == 2
    assert db.store[b"a"] == b"1" and db.store[b"b"] == b"2"

def test_writebatch_works_with_put_cf(monkeypatch):
    import rocks_shim as rs
    monkeypatch.setattr(rs, "WriteBatch", WB_put_cf, raising=True)
    db = DB_write_batch_with_put_cf()
    wrote = write_batch_to_db(db, {"x": b"9"})
    assert wrote == 1
    assert db.store[b"x"] == b"9"
