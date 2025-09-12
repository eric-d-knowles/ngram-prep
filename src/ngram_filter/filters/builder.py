# ngram_filter/filters/builder.py
from typing import Optional, Callable, Iterable, Any
import builtins
from ..config import FilterConfig
from .core_cy import process_tokens as _impl_process_tokens  # bytes -> bytes

def _to_bytes_set(s: Optional[Iterable[Any]]):
    """
    Normalize an iterable of {str|bytes|bytearray} into a set[bytes] (ASCII).
    Returns None if s is None. Empty iterables return an empty set.
    """
    if s is None:
        return None
    try:
        it = iter(s)
    except TypeError:
        raise TypeError("stop_set/vocab_set must be an iterable of str/bytes/bytearray")

    out: set[bytes] = set()
    for w in it:
        if isinstance(w, (builtins.bytes, builtins.bytearray)):
            out.add(builtins.bytes(w))
        elif isinstance(w, str):
            out.add(w.encode("ascii"))
        else:
            raise TypeError(f"Unsupported token type: {type(w).__name__} "
                            "(expected str|bytes|bytearray)")
    return out

def build_processor(cfg: FilterConfig) -> Callable[[bytes], Optional[bytes]]:
    """
    Build a bytes-only processor: (bytes ngram) -> Optional[bytes].
    Returns None to drop the ngram.
    """
    stop_set_b = _to_bytes_set(cfg.stop_set)

    # vocab can be:
    #  - an MMapVocab (preferred; must implement __contains__(bytes))
    #  - a small in-memory set of tokens (iterable -> set[bytes])
    vocab_obj = cfg.vocab_view
    if vocab_obj is None:
        # Optionally support tiny vocab passed as iterable on cfg (not present now, but harmless)
        # vocab_obj = _to_bytes_set(getattr(cfg, "vocab_set", None))
        pass

    outbuf = bytearray()  # reused per-processor

    def _processor(token_b: bytes) -> Optional[bytes]:
        if not isinstance(token_b, (builtins.bytes, builtins.bytearray)):
            raise TypeError("token_b must be bytes or bytearray")
        if isinstance(token_b, builtins.bytearray):
            token_b = builtins.bytes(token_b)

        outbuf.clear()
        out = _impl_process_tokens(
            token_b,
            opt_lower=cfg.opt_lower,
            opt_alpha=cfg.opt_alpha,
            opt_shorts=cfg.opt_shorts,
            opt_stops=cfg.opt_stops,
            opt_lemmas=cfg.opt_lemmas,
            min_len=cfg.min_len,
            stop_set=stop_set_b,
            lemma_gen=cfg.lemma_gen,
            vocab_set=vocab_obj,     # <-- previously undefined variable; now the mmap view or None
            outbuf=outbuf,
        )
        return out or None

    return _processor
