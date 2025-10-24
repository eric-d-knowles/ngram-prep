#!/usr/bin/env python3
"""
rocksdb_compact_watch.py

Cross-process monitor for RocksDB compactions using your Python shim.

Signals used (robust when monitoring a DB compacted by another process):
- rocksdb.levelstats parsing (compactions-in-progress per level)
- Filesystem activity (recent writes to *.sst / *.sst.tmp)
- Deltas in file counts/sizes between ticks
- LOG tail scan (EVENT_LOG_v1 compaction_started, [JOB ...] Compacting, Manual compaction, table_file_*)

We also show DB-wide props (num-running-compactions/flushes, pending, pending-bytes)
and cfstats if available, but those often read as 0 from a separate read-only opener.

Examples:
  ./rocksdb_compact_watch.py /path/to/db
  ./rocksdb_compact_watch.py /path/to/db --watch --interval 5 --log-path /path/to/LOG
  ./rocksdb_compact_watch.py /path/to/db --json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -------------------------------------------------------------------------
# Shim import
# -------------------------------------------------------------------------
try:
    import rocks_shim
except Exception as e:
    print(f"ERROR: Failed to import rocks_shim: {e}", file=sys.stderr)
    sys.exit(1)

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
LEVELSTATS_LINE = re.compile(
    r"^Level\s+(\d+)\s+\|\s+Files:\s+(\d+).*?Compactions\s+in\s+progress:\s+(\d+)",
    re.I,
)

CFSTATS_RUNNING_RE   = re.compile(r"Compactions running:\s*(\d+)", re.I)
CFSTATS_PENDING_RE   = re.compile(r"Compactions pending:\s*(\d+)", re.I)
CFSTATS_FLUSHING_RE  = re.compile(r"Memtables flushing:\s*(\d+)", re.I)

def _fmt_gib(nbytes: int) -> str:
    return f"{(nbytes or 0) / (1024**3):.2f} GiB"

def _safe_int(v, default: int = 0) -> int:
    """Robust int parsing; treats None/'' as unavailable, not zero."""
    if v in (None, "", "N/A"):
        return default
    try:
        return int(v)
    except Exception:
        try:
            m = re.search(r"(\d+)", str(v))
            return int(m.group(1)) if m else default
        except Exception:
            return default

def _collect_sst_sizes(db_path: Path) -> List[int]:
    sizes: List[int] = []
    for g in (db_path.glob("*.sst"), (db_path / "archive").glob("*.sst")):
        for p in g:
            try:
                sizes.append(p.stat().st_size)
            except (FileNotFoundError, PermissionError):
                pass
    return sizes

def _count_tmp_ssts(db_path: Path) -> int:
    cnt = 0
    for p in db_path.glob("*.sst.tmp"):
        try:
            p.stat()
            cnt += 1
        except FileNotFoundError:
            pass
    return cnt

def _histogram_mib(sizes_bytes: List[int], mb_bucket: int) -> Counter:
    def to_bucket(b: int) -> int:
        mib = b / (1024 * 1024)
        return int(math.floor(mib / mb_bucket) * mb_bucket)
    return Counter(to_bucket(b) for b in sizes_bytes)

# -------------------------------------------------------------------------
# LOG tail (patched for your patterns)
# -------------------------------------------------------------------------
def _tail_log(log_path: Optional[Path], lines: int = 400) -> Dict[str, int]:
    """
    Scan the tail of the LOG for common compaction/flush markers used by RocksDB:
    - EVENT_LOG_v1 "compaction_started"
    - "[JOB N] Compacting ..." lines
    - "Manual compaction ..." lines
    - table_file_creation / table_file_deletion
    - Flush markers
    """
    if not log_path:
        return {}
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = 128 * 1024  # scan a larger window to catch a few events
            data = b""
            while size > 0 and data.count(b"\n") <= lines:
                read = min(block, size)
                size -= read
                f.seek(size)
                data = f.read(read) + data
            text = data.decode("utf-8", "replace")
    except Exception:
        return {}

    compaction_started = len(re.findall(r'event":\s*"compaction_started"', text))
    job_compacting     = len(re.findall(r'\[JOB\s+\d+\]\s+Compacting\b', text))
    manual_compaction  = len(re.findall(r'\bManual compaction\b', text))
    table_create       = len(re.findall(r'event":\s*"table_file_creation"', text))
    table_delete       = len(re.findall(r'event":\s*"table_file_deletion"', text))
    flush_markers      = len(re.findall(r'\bFlush\b|\bflushing\b', text))

    comp_hits = compaction_started + job_compacting + manual_compaction

    return {
        "log_compaction_started": compaction_started,
        "log_job_compacting": job_compacting,
        "log_manual_compaction": manual_compaction,
        "log_table_creations": table_create,
        "log_table_deletions": table_delete,
        "log_flush_mentions": flush_markers,
        "log_compaction_hits": comp_hits,
    }

# -------------------------------------------------------------------------
# Property fetch & parsing
# -------------------------------------------------------------------------
def _get(db, name: str):
    """Get property; normalize empty string to None."""
    try:
        val = db.get_property(name)
        if isinstance(val, str) and val.strip() == "":
            return None
        return val
    except Exception:
        return None

def _parse_levelstats(db) -> Tuple[Dict[int, int], int]:
    """
    Returns (files_by_level, compactions_in_progress_sum) using rocksdb.levelstats.
    This is metadata-based and works cross-process in most builds.
    """
    txt = _get(db, "rocksdb.levelstats") or ""
    files_by_level: Dict[int, int] = {}
    comp_in_prog = 0
    for line in txt.splitlines():
        m = LEVELSTATS_LINE.search(line)
        if m:
            L = int(m.group(1))
            files_by_level[L] = int(m.group(2))
            comp_in_prog += _safe_int(m.group(3), 0)
    return files_by_level, comp_in_prog

def _parse_cfstats(db) -> Dict[str, int]:
    """Parse rocksdb.cfstats (requires stats enabled on that instance)."""
    txt = _get(db, "rocksdb.cfstats") or ""
    running = sum(int(m.group(1)) for m in CFSTATS_RUNNING_RE.finditer(txt))
    pending = sum(int(m.group(1)) for m in CFSTATS_PENDING_RE.finditer(txt))
    flushing = sum(int(m.group(1)) for m in CFSTATS_FLUSHING_RE.finditer(txt))
    return {
        "cf_compactions_running": running,
        "cf_compactions_pending": pending,
        "cf_memtables_flushing": flushing,
    }

def _gather_db_props(db) -> Dict[str, int]:
    return {
        "num_running_compactions": _safe_int(_get(db, "rocksdb.num-running-compactions")),
        "num_running_flushes":     _safe_int(_get(db, "rocksdb.num-running-flushes")),
        "compaction_pending":      _safe_int(_get(db, "rocksdb.compaction-pending")),
        "est_pending_comp_bytes":  _safe_int(_get(db, "rocksdb.estimate-pending-compaction-bytes")),
    }

def _fs_recent_activity(db_path: Path, window_sec: int = 20) -> Dict[str, int]:
    now = time.time()
    recent = {"recent_sst_writes": 0, "recent_tmp_writes": 0}
    try:
        for p in db_path.glob("*.sst"):
            try:
                if now - p.stat().st_mtime <= window_sec:
                    recent["recent_sst_writes"] += 1
            except FileNotFoundError:
                pass
        for p in db_path.glob("*.sst.tmp"):
            try:
                if now - p.stat().st_mtime <= window_sec:
                    recent["recent_tmp_writes"] += 1
            except FileNotFoundError:
                pass
    except Exception:
        pass
    return recent

# -------------------------------------------------------------------------
# Snapshot & status
# -------------------------------------------------------------------------
def compaction_snapshot(
    db,
    db_path: Path,
    max_level: int,
    mb_bucket: int,
    log_path: Optional[Path],
    log_lines: int,
) -> Dict:
    sizes = _collect_sst_sizes(db_path)
    total_bytes = sum(sizes)
    hist = _histogram_mib(sizes, mb_bucket)

    # Prefer metadata/FS signals for cross-process monitoring
    files_by_level, lvl_comp_in_prog = _parse_levelstats(db)
    db_props = _gather_db_props(db)
    cfstats = _parse_cfstats(db)  # may be zeros cross-process; keep for context
    tmp_sst = _count_tmp_ssts(db_path)
    fs_recent = _fs_recent_activity(db_path)
    log_counts = _tail_log(log_path, lines=log_lines)

    # Fill shallow levels to max_level for consistent output
    for L in range(max_level):
        files_by_level.setdefault(L, 0)

    snap = {
        "db_path": str(db_path),
        "total_sst_bytes": total_bytes,
        "total_sst_human": _fmt_gib(total_bytes),
        "files_per_level": OrderedDict((f"L{L}", files_by_level[L]) for L in sorted(files_by_level)),
        "sst_size_modes_mib": OrderedDict((int(m), int(c)) for m, c in sorted(hist.items())),
        "tmp_sst_files": tmp_sst,
        "levelstats_compactions_in_progress": lvl_comp_in_prog,
        **db_props,
        **cfstats,
        **fs_recent,
        **log_counts,
    }
    snap["status"] = _infer_status(snap)
    return snap

def _infer_status(s: Dict) -> str:
    """
    Cross-process activity heuristic:
    - Strong signals: levelstats compactions in progress, recent *.sst.tmp or *.sst writes, LOG compaction hits
    - Secondary: estimate-pending-compaction-bytes
    - Tertiary (often 0 cross-process): num-running-* and cfstats
    """
    active = 0
    # Strong
    active += 1 if s.get("levelstats_compactions_in_progress", 0) > 0 else 0
    active += 1 if s.get("recent_tmp_writes", 0) > 0 else 0
    active += 1 if s.get("recent_sst_writes", 0) > 0 else 0
    active += 1 if s.get("log_compaction_hits", 0) > 0 else 0
    # Secondary
    active += 1 if s.get("est_pending_comp_bytes", 0) > 0 else 0
    # Tertiary
    active += 1 if s.get("num_running_compactions", 0) > 0 else 0
    active += 1 if s.get("compaction_pending", 0) > 0 else 0
    active += 1 if s.get("cf_compactions_running", 0) > 0 else 0
    active += 1 if s.get("cf_compactions_pending", 0) > 0 else 0

    if active >= 2:
        return "compaction_active"
    if active == 1:
        return "maybe_active"
    return "quiescent"

def _print_snapshot_human(s: Dict, no_color: bool = False, delta: Optional[Dict] = None) -> None:
    def c(text, code):
        return text if no_color else f"\x1b[{code}m{text}\x1b[0m"

    print(c("\n=== RocksDB Compaction Snapshot ===", "1"))
    print(f"DB path: {s['db_path']}")
    print(f"Total SST size: {s['total_sst_human']}")
    if s["files_per_level"]:
        print("Levels: " + ", ".join(f"{lvl}:{cnt}" for lvl, cnt in s["files_per_level"].items()))
    else:
        print("Levels: (none)")

    print("\nActivity signals:")
    print(f"  Levelstats: compactions_in_progress={s.get('levelstats_compactions_in_progress',0)}")
    print(f"  DB props  : running_compactions={s.get('num_running_compactions',0)} "
          f"running_flushes={s.get('num_running_flushes',0)} "
          f"pending={s.get('compaction_pending',0)} "
          f"est_pending_bytes={s.get('est_pending_comp_bytes',0)}")
    print(f"  CF stats  : compactions_running={s.get('cf_compactions_running',0)} "
          f"compactions_pending={s.get('cf_compactions_pending',0)} "
          f"memtables_flushing={s.get('cf_memtables_flushing',0)}")
    print(f"  FS clues  : tmp_sst_files={s.get('tmp_sst_files',0)}")
    print(f"  FS recent : sst_writes(last 20s)={s.get('recent_sst_writes',0)} "
          f"tmp_writes(last 20s)={s.get('recent_tmp_writes',0)}")
    if "log_compaction_hits" in s:
        print(f"  LOG tail  : compaction_hits={s.get('log_compaction_hits',0)} "
              f"(started={s.get('log_compaction_started',0)}, "
              f"job_compacting={s.get('log_job_compacting',0)}, "
              f"manual={s.get('log_manual_compaction',0)}; "
              f"creates={s.get('log_table_creations',0)}, deletes={s.get('log_table_deletions',0)}, "
              f"flush={s.get('log_flush_mentions',0)})")

    if delta:
        print("\nDeltas since last tick:")
        for k, v in delta.items():
            print(f"  {k}: {v:+d}")

    print("\nSST size modes (MiB):")
    if s["sst_size_modes_mib"]:
        for mib, cnt in s["sst_size_modes_mib"].items():
            print(f"  ~{mib:>5} MiB : {cnt:>6} files")
    else:
        print("  (no .sst files found)")

    status = s["status"]
    if status == "compaction_active":
        print("\n" + c("Status: compactions active (multiple independent signals).", "33"))
    elif status == "maybe_active":
        print("\n" + c("Status: at least one active signal detected — likely active.", "36"))
    else:
        print("\n" + c("Status: quiescent (no active signals).", "32"))

# -------------------------------------------------------------------------
# DB open & runners
# -------------------------------------------------------------------------
def _open_db(db_path: Path):
    # read-only opener via your shim
    return rocks_shim.DB.open(str(db_path), read_only=True)

def _delta(prev: Dict, curr: Dict) -> Dict[str, int]:
    d: Dict[str, int] = {}
    prev_levels = prev.get("files_per_level", {})
    curr_levels = curr.get("files_per_level", {})
    for lvl in set(prev_levels.keys()) | set(curr_levels.keys()):
        p = int(prev_levels.get(lvl, 0))
        q = int(curr_levels.get(lvl, 0))
        if p != q:
            d[f"files_{lvl}"] = q - p
    if prev.get("tmp_sst_files", 0) != curr.get("tmp_sst_files", 0):
        d["tmp_sst_files"] = curr.get("tmp_sst_files", 0) - prev.get("tmp_sst_files", 0)
    return d

def run_once(db_path: Path, max_level: int, mb_bucket: int, json_out: bool,
             no_color: bool, log_path: Optional[Path], log_lines: int) -> int:
    try:
        db = _open_db(db_path)
    except Exception as e:
        print(f"ERROR: opening DB at {db_path}: {e}", file=sys.stderr)
        return 2
    try:
        snap = compaction_snapshot(db, db_path, max_level, mb_bucket, log_path, log_lines)
        if json_out:
            print(json.dumps(snap, indent=2))
        else:
            _print_snapshot_human(snap, no_color=no_color)
        return 0
    finally:
        try:
            db.close()
        except Exception:
            pass

def run_watch(db_path: Path, max_level: int, mb_bucket: int, interval: int, rounds: Optional[int],
              json_out: bool, no_color: bool, log_path: Optional[Path], log_lines: int) -> int:
    i = 0
    prev = None
    try:
        while rounds is None or i < rounds:
            try:
                db = _open_db(db_path)
            except Exception as e:
                print(f"ERROR: opening DB at {db_path}: {e}", file=sys.stderr)
                return 2
            try:
                snap = compaction_snapshot(db, db_path, max_level, mb_bucket, log_path, log_lines)
            finally:
                try:
                    db.close()
                except Exception:
                    pass

            if json_out:
                print(json.dumps(snap, indent=2))
            else:
                delta = _delta(prev, snap) if prev else None
                _print_snapshot_human(snap, no_color=no_color, delta=delta)
            prev = snap
            i += 1
            time.sleep(interval)
        return 0
    except KeyboardInterrupt:
        return 130

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monitor RocksDB compaction state (cross-process signals) via rocks_shim."
    )
    p.add_argument("db_path", type=Path, help="Path to the RocksDB database directory")
    p.add_argument("--max-level", type=int, default=8, help="Max level index to show (default: 8)")
    p.add_argument("--mb-bucket", type=int, default=1, help="Histogram bucket width in MiB (default: 1)")
    p.add_argument("--watch", action="store_true", help="Refresh periodically")
    p.add_argument("--interval", type=int, default=30, help="Refresh interval seconds (default: 30)")
    p.add_argument("--rounds", type=int, default=None, help="Iterations for --watch (default: unlimited)")
    p.add_argument("--json", dest="json_out", action="store_true", help="Emit JSON instead of human-readable text")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    p.add_argument("--log-path", type=Path, default=None, help="Path to RocksDB LOG (optional)")
    p.add_argument("--log-lines", type=int, default=400, help="How many lines to scan from LOG tail (default: 400)")
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not args.db_path.exists():
        print(f"ERROR: DB path does not exist: {args.db_path}", file=sys.stderr)
        return 2

    if args.watch:
        return run_watch(
            db_path=args.db_path,
            max_level=args.max_level,
            mb_bucket=args.mb_bucket,
            interval=args.interval,
            rounds=args.rounds,
            json_out=args.json_out,
            no_color=args.no_color,
            log_path=args.log_path,
            log_lines=args.log_lines,
        )
    else:
        return run_once(
            db_path=args.db_path,
            max_level=args.max_level,
            mb_bucket=args.mb_bucket,
            json_out=args.json_out,
            no_color=args.no_color,
            log_path=args.log_path,
            log_lines=args.log_lines,
        )

if __name__ == "__main__":
    sys.exit(main())
