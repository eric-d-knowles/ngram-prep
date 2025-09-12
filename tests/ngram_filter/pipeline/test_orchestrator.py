# tests/ngram_filter/pipeline/test_orchestrator.py
from __future__ import annotations

import multiprocessing as mp
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pytest

# import the module that will contain build_processed_db_sharded
import ngram_filter.pipeline.old.orchestrator as orch_mod
from ngram_filter.pipeline.old.routing import Router


@pytest.fixture(autouse=True)
def mp_fork():
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    yield


# ----------------- Dummy configs -----------------
@dataclass
class PipelineCfg:
    src_db: Path
    dst_db: Path
    tmp_dir: Path
    outer_writers: int = 3
    readers: int = 2
    queue_maxsize: int = 100
    progress_every_s: int = 0
    # old names mapped to per-lane thresholds inside orchestrator
    max_items_per_bucket: int = 0
    max_bytes_per_bucket: int = 0
    # new router knobs (optional in orchestrator)
    inner_lanes: int = 2
    router_seed: int = 0

@dataclass
class FilterCfg:
    # whatever your reader expects; unused by the stub
    dummy: int = 0


# ----------------- Stubs & spies -----------------
@contextmanager
def _noop_open_db(*args, **kwargs):
    yield object()  # dummy DB


def test_orchestrator_wires_processes_and_calls_ingest(tmp_path, monkeypatch):
    """
    Drives the refactor: verifies that build_processed_db_sharded
    - spawns one writer per shard (outer_writers)
    - spawns readers
    - routes items to the right shard queues
    - sends sentinels, collects writer summaries
    - calls ingest_shards_streaming with (dst_db, shards_root)
    """
    # Make mp context predictable
    monkeypatch.setattr(orch_mod, "_mp_ctx", lambda: mp.get_context("fork"))

    # Router distribution we will rely on (same as orchestrator will create)
    router = Router(num_shards=3, inner_lanes=2, seed=0)

    # Shared state (Manager dict) visible to child processes
    mgr = mp.Manager()
    per_shard_items: Dict[int, List[bytes]] = mgr.dict({0: mgr.list(), 1: mgr.list(), 2: mgr.list()})
    ingest_calls = mgr.list()

    # Stub reader: push a few keys into the proper shard queues via router
    def fake_reader_worker(**kwargs):
        shard_queues = kwargs["shard_queues"]
        router = kwargs["router"]
        for k in [b"a0", b"b1", b"c2", b"a3"]:  # deterministic small sample
            outer = router.route_outer(k)
            shard_queues[outer].put((k, b"V"))
        # done

    # Stub writer: drain queue until None, record keys, send summary via conn
    def fake_writer_entry(*, shard_id, q, db_root, router, conn, cfg, counters, **_):
        while True:
            item = q.get()
            if item is None:
                break
            k, v = item
            per_shard_items[shard_id].append(k)
        try:
            conn.send({"shard_id": shard_id, "ok": True})
        finally:
            conn.close()

    # Spy ingest: just record the args
    def fake_ingest(dst_db_path: Path, shards_root: Path):
        ingest_calls.append((str(dst_db_path), str(shards_root)))

    # Patch orchestrator dependencies
    monkeypatch.setattr(orch_mod, "reader_worker", fake_reader_worker)
    monkeypatch.setattr(orch_mod, "writer_entry", fake_writer_entry)
    monkeypatch.setattr(orch_mod, "ingest_shards_streaming", fake_ingest)

    # Make sure the orchestrator builds the same Router config (3 shards, 2 lanes, seed 0)
    # by giving a PipelineCfg that matches those numbers.
    pcfg = PipelineCfg(
        src_db=tmp_path / "src.db",
        dst_db=tmp_path / "dst.db",
        tmp_dir=tmp_path / "tmp",
        outer_writers=router.num_shards,
        readers=1,
        inner_lanes=router.inner_lanes,
        router_seed=router.seed,
    )
    fcfg = FilterCfg()

    # Act: run the orchestrator
    orch_mod.build_processed_db_sharded(pcfg, fcfg)

    # Assert routing happened and writers drained correctly
    # Compute expected shard for each key via the same router
    expected = {0: [], 1: [], 2: []}
    for k in [b"a0", b"b1", b"c2", b"a3"]:
        expected[router.route_outer(k)].append(k)

    observed = {sid: list(per_shard_items[sid]) for sid in [0, 1, 2]}
    assert observed == expected

    # Assert ingest was called once with the right paths
    assert len(ingest_calls) == 1
    called_dst, called_root = ingest_calls[0]
    assert called_dst == str(pcfg.dst_db.resolve())
    assert called_root.endswith("shards_stage")
