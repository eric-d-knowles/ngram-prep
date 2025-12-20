# ngram_filter/filters/shared_vocab.py
from __future__ import annotations
from pathlib import Path
from array import array
import mmap
import io
from typing import Iterable, Union, Optional
import struct

_U64 = struct.Struct("<Q")

def _iter_clean_lines(path: Path) -> Iterable[bytes]:
    # Read tokens only (left field). Lines may be "token<TAB>freq" or "token freq".
    with path.open("rt", encoding="utf-8", errors="strict", newline="") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "\t" in s:
                tok = s.split("\t", 1)[0]
            else:
                # fall back to first whitespace-separated field
                tok = s.split(None, 1)[0]
            if tok:
                yield tok.encode("utf-8")


def build_vocab_index(vocab_txt: Union[str, Path], out_prefix: Union[str, Path]) -> tuple[Path, Path]:
    """
    Build a compact, shareable, exact index:
      out_prefix.idx : array('Q') of length N+1 (byte offsets)
      out_prefix.lex : concatenated UTF-8 tokens, no separators
    """
    vocab_txt = Path(vocab_txt)
    out_prefix = Path(out_prefix)
    idx_path = out_prefix.with_suffix(".idx")
    lex_path = out_prefix.with_suffix(".lex")

    # Load, dedup, sort as bytes
    tokens = sorted(set(_iter_clean_lines(vocab_txt)))
    n = len(tokens)

    # Write *.lex and *.idx (N+1 offsets; last is total length)
    offsets = array("Q", [0] * (n + 1))
    total = 0

    # Stream to disk without holding a giant buffer
    with lex_path.open("wb") as lf:
        for i, tok in enumerate(tokens):
            lf.write(tok)
            total += len(tok)
            offsets[i + 1] = total

    with idx_path.open("wb") as ifp:
        offsets.tofile(ifp)

    return idx_path, lex_path


class MMapVocab:
    """
    Memory-mapped, sorted vocabulary with binary search membership.
    Expects:
      <prefix>.idx : uint64 offsets into <prefix>.lex, length M+1 (sentinel at end)
      <prefix>.lex : concatenated token bytes (no separators required)
    """
    def __init__(self, prefix: str | Path):
        prefix = Path(prefix)
        idx_path = prefix.with_suffix(".idx")
        lex_path = prefix.with_suffix(".lex")

        # Open + mmap (keep file handles to prevent GC closing mmaps)
        self._idx_f = idx_path.open("rb", buffering=0)
        self._lex_f = lex_path.open("rb", buffering=0)
        self._idx_mm = mmap.mmap(self._idx_f.fileno(), 0, access=mmap.ACCESS_READ)
        self._lex_mm = mmap.mmap(self._lex_f.fileno(), 0, access=mmap.ACCESS_READ)

        self._mv_idx = memoryview(self._idx_mm)
        self._mv_lex = memoryview(self._lex_mm)

        # Number of entries (assumes sentinel offset exists)
        n_offsets = len(self._mv_idx) // _U64.size
        if n_offsets < 2:
            self._count = 0
        else:
            self._count = n_offsets - 1  # <-- define _count

        # Bind unpacker once
        self._unpack64 = _U64.unpack_from

    def __len__(self) -> int:
        return self._count

    def _slice(self, i: int) -> memoryview:
        # Return memoryview of token i (no allocations)
        off0 = self._unpack64(self._mv_idx, i * 8)[0]
        off1 = self._unpack64(self._mv_idx, (i + 1) * 8)[0]
        return self._mv_lex[off0:off1]

    def __contains__(self, key) -> bool:
        # Normalize key to bytes once
        if isinstance(key, (bytes, bytearray)):
            key_b = bytes(key)
        elif isinstance(key, memoryview):
            key_b = key.tobytes()
        else:
            return False  # or raise TypeError

        lo = 0
        hi = self._count - 1
        to_b = memoryview.tobytes  # micro-opt to avoid attribute lookups

        while lo <= hi:
            mid = (lo + hi) >> 1
            piv_b = to_b(self._slice(mid))  # bytes for lexicographic compare

            if piv_b < key_b:
                lo = mid + 1
            elif piv_b > key_b:
                hi = mid - 1
            else:
                return True

        return False

    def close(self) -> None:
        try:
            self._mv_idx.release()
            self._mv_lex.release()
        except Exception:
            pass
        try:
            self._idx_mm.close()
            self._lex_mm.close()
        except Exception:
            pass
        try:
            self._idx_f.close()
            self._lex_f.close()
        except Exception:
            pass
