# ngram_filter/filters/processor_factory.py
"""Factory for building ngram filter processors."""

from __future__ import annotations

from typing import Optional, Callable, Iterable, Any, Protocol
import builtins
from ..config import FilterConfig
from .core_cy import process_tokens as _impl_process_tokens  # bytes -> bytes


class ProcessorProtocol(Protocol):
    """Protocol for ngram processor functions."""
    def __call__(self, token: bytes) -> Optional[bytes]: ...

def _to_bytes_set(s: Optional[Iterable[Any]]):
    """
    Normalize an iterable of {str|bytes|bytearray} into a set[bytes] (UTF-8).
    Returns None if s is None. Empty iterables return an empty set.
    """
    if s is None:
        return None
    try:
        it = iter(s)
    except TypeError:
        raise TypeError("stop_set must be an iterable of str/bytes/bytearray")

    out: set[bytes] = set()
    for w in it:
        if isinstance(w, (builtins.bytes, builtins.bytearray)):
            out.add(builtins.bytes(w))
        elif isinstance(w, str):
            out.add(w.encode("utf-8"))
        else:
            raise TypeError(f"Unsupported token type: {type(w).__name__} "
                            "(expected str|bytes|bytearray)")
    return out

def build_processor(cfg: FilterConfig) -> ProcessorProtocol:
    """
    Build a bytes-only processor: (bytes ngram) -> Optional[bytes].
    Returns None to drop the ngram.
    """
    stop_set_b = _to_bytes_set(cfg.stop_set)
    always_include_b = _to_bytes_set(cfg.always_include)

    outbuf = bytearray()  # reused per-processor

    def _processor(token_b: bytes) -> Optional[bytes]:
        if not isinstance(token_b, (builtins.bytes, builtins.bytearray)):
            raise TypeError("token_b must be bytes or bytearray")
        if isinstance(token_b, builtins.bytearray):
            token_b = builtins.bytes(token_b)

        outbuf.clear()
        out = _impl_process_tokens(
            token_b,
            opt_lower=cfg.lowercase,
            opt_alpha=cfg.alpha_only,
            opt_ascii_alpha_only=cfg.ascii_alpha_only,
            opt_shorts=cfg.filter_short,
            opt_stops=cfg.filter_stops,
            opt_lemmas=cfg.apply_lemmatization,
            min_len=cfg.min_len,
            stop_set=stop_set_b,
            lemma_gen=cfg.lemma_gen,
            whitelist=cfg.whitelist,
            always_include=always_include_b,
            outbuf=outbuf,
        )
        return out or None

    return _processor