# tests/test_builder.py
import builtins
import types
import pytest

# Module under test
import ngram_filter.filters.builder as builder

# A tiny stand-in FilterConfig to avoid importing the whole project
class DummyFilterConfig:
    def __init__(
        self,
        *,
        stop_set=None,
        vocab_set=None,
        opt_lower=False,
        opt_alpha=False,
        opt_shorts=False,
        opt_stops=False,
        opt_lemmas=False,
        min_len=0,
        lemma_gen=None,
    ):
        self.stop_set = stop_set
        self.vocab_set = vocab_set
        self.opt_lower = opt_lower
        self.opt_alpha = opt_alpha
        self.opt_shorts = opt_shorts
        self.opt_stops = opt_stops
        self.opt_lemmas = opt_lemmas
        self.min_len = min_len
        self.lemma_gen = lemma_gen


# ----------------------------- _to_bytes_set -----------------------------

def test_to_bytes_set_none_passthrough():
    assert builder._to_bytes_set(None) is None

def test_to_bytes_set_empty_passthrough():
    assert builder._to_bytes_set(set()) == set()

def test_to_bytes_set_of_strs_becomes_bytes():
    out = builder._to_bytes_set({"a", "b"})
    assert out == {b"a", b"b"}
    assert all(isinstance(x, builtins.bytes) for x in out)

def test_to_bytes_set_of_bytes_normalized():
    out = builder._to_bytes_set([b"a", bytearray(b"b")])  # list, not set
    assert out == {b"a", b"b"}
    assert all(isinstance(x, bytes) for x in out)

def test_to_bytes_set_mixed_str_and_bytes():
    out = builder._to_bytes_set(["a", b"b", bytearray(b"c")])
    assert out == {b"a", b"b", b"c"}


# ----------------------------- build_processor -----------------------------

def test_processor_calls_impl_with_expected_args(monkeypatch):
    calls = []

    def fake_impl(token_b, **kwargs):
        # Capture the call for assertions
        calls.append((token_b, kwargs))
        # Return some non-empty bytes so processor returns bytes
        return b"OK"

    # monkeypatch the cython function alias used by builder
    monkeypatch.setattr(builder, "_impl_process_tokens", fake_impl, raising=True)

    cfg = DummyFilterConfig(
        stop_set={"the", "a"},
        vocab_set={"cat"},
        opt_lower=True, opt_alpha=True, opt_shorts=False,
        opt_stops=True, opt_lemmas=False,
        min_len=2,
        lemma_gen=None,
    )

    proc = builder.build_processor(cfg)
    out = proc(b"THE CAT")

    # Returned bytes from fake impl should come through
    assert out == b"OK"

    # Exactly one call recorded
    assert len(calls) == 1
    token_b, kwargs = calls[0]

    assert token_b == b"THE CAT"
    # Flags/params forwarded correctly
    assert kwargs["opt_lower"] is True
    assert kwargs["opt_alpha"] is True
    assert kwargs["opt_shorts"] is False
    assert kwargs["opt_stops"] is True
    assert kwargs["opt_lemmas"] is False
    assert kwargs["min_len"] == 2
    assert kwargs["lemma_gen"] is None

    # stop_set/vocab_set must have been converted to bytes
    assert kwargs["stop_set"] == {b"the", b"a"} or kwargs["stop_set"] == {b"a", b"the"}
    assert kwargs["vocab_set"] == {b"cat"}


def test_processor_returns_none_on_empty_bytes(monkeypatch):
    def fake_impl(token_b, **kwargs):
        return b""  # cause the wrapper to return None

    monkeypatch.setattr(builder, "_impl_process_tokens", fake_impl, raising=True)

    cfg = DummyFilterConfig()
    proc = builder.build_processor(cfg)

    assert proc(b"anything") is None


def test_outbuf_is_reused_across_calls(monkeypatch):
    # We’ll record the id() of the outbuf object across calls.
    # The builder module keeps a single bytearray per-processor; we want it stable.
    seen_ids = []

    def fake_impl(token_b, **kwargs):
        outbuf = kwargs["outbuf"]
        # Ensure it is a bytearray and track identity
        assert isinstance(outbuf, bytearray)
        seen_ids.append(id(outbuf))
        # Write something into outbuf to mimic behavior (not required, but realistic)
        outbuf.extend(b"X")
        return b"Y"

    monkeypatch.setattr(builder, "_impl_process_tokens", fake_impl, raising=True)

    cfg = DummyFilterConfig()
    proc = builder.build_processor(cfg)

    _ = proc(b"one")
    _ = proc(b"two")
    _ = proc(b"three")

    # Same object reused
    assert len(set(seen_ids)) == 1


def test_processor_requires_bytes_input(monkeypatch):
    def fake_impl(token_b, **kwargs):
        return b"OK"

    monkeypatch.setattr(builder, "_impl_process_tokens", fake_impl, raising=True)

    cfg = DummyFilterConfig()
    proc = builder.build_processor(cfg)

    with pytest.raises(TypeError):
        # builder.proc signature is annotated bytes; passing str should explode before fake_impl
        proc("not-bytes")  # type: ignore[arg-type]
