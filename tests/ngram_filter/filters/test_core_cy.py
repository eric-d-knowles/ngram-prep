# tests/filters/test_core_cy.py
import pytest

try:
    # Compiled extension (pyx) should expose these
    from ngram_filter.filters.core_cy import (
        process_tokens,
        SENTINEL_B,
    )
except Exception as e:  # pragma: no cover
    process_tokens = None
    SKIP_REASON = f"core_cy extension not importable: {e!r}"

pytestmark = pytest.mark.skipif(
    process_tokens is None,
    reason=globals().get("SKIP_REASON", "core_cy extension not built"),
)


# -------------------------- process_tokens ---------------------

def test_passthrough_no_options():
    out = process_tokens(b"Dog_NOUN runs_VERB fast_ADV")
    # With all options defaulting to False and no vocab/stops, returns base forms
    # Tags are removed when recognized in the G-tag set
    assert out == b"Dog runs fast"

def test_lowercasing_only():
    out = process_tokens(b"Dog_NOUN runs_VERB fast_ADV", opt_lower=True)
    assert out == b"dog runs fast"

def test_vocabulary_filter_bytes():
    vocab = {b"dog", b"runs", b"fast"}  # bytes vocab
    # With opt_lower, base tokens are lowercased before lookup
    out = process_tokens(
        b"Dog_NOUN runs_VERB fast_ADV",
        opt_lower=True,
        vocab_set=vocab,
    )
    assert out == b"dog runs fast"

    # If one token is out-of-vocab, it becomes <UNK>
    out = process_tokens(
        b"Dog_NOUN runs_VERB zooms_VERB",
        opt_lower=True,
        vocab_set=vocab,
    )
    # Entire line doesn't drop unless ALL tokens turn to UNK
    assert out == b"dog runs " + SENTINEL_B

def test_alpha_filter_blocks_non_ascii_and_punct():
    # Å (non-ASCII) and 'cat!' (punct) should be UNK if opt_alpha
    out = process_tokens(b"\xC3\x85_NOUN cat!_NOUN ok_NOUN", opt_alpha=True)
    assert out == SENTINEL_B + b" " + SENTINEL_B + b" ok"

def test_short_word_filter():
    out = process_tokens(b"aa_NOUN bee_NOUN see_NOUN", opt_shorts=True, min_len=3)
    # 'aa' is too short -> <UNK>
    assert out == SENTINEL_B + b" bee see"

def test_stopword_filter_bytes():
    stops = {b"the", b"and"}
    out = process_tokens(
        b"the_DET big_ADJ cat_NOUN and_CONJ dog_NOUN",
        opt_stops=True,
        stop_set=stops,
        opt_lower=True,
    )
    assert out == SENTINEL_B + b" big cat " + SENTINEL_B + b" dog"

def test_lemmas_when_pos_mapped_and_ascii():
    class DummyLemma:
        def lemmatize(self, s, pos="n"):
            # simple rule: VERB -> base 'run', NOUN -> 'dog', ADJ -> 'good', ADV -> 'well'
            return {"v": "run", "n": "dog", "a": "good", "r": "well"}.get(pos, s)

    out = process_tokens(
        b"Dogs_NOUN running_VERB good_ADJ well_ADV",
        opt_lemmas=True,
        lemma_gen=DummyLemma(),
        opt_alpha=True,      # enforces ASCII isalpha on output
        opt_lower=True,
    )
    assert out == b"dog run good well"

def test_all_tokens_become_unk_returns_empty():
    vocab = {b"only"}  # excludes everything in the input
    out = process_tokens(
        b"foo_NOUN bar_VERB",
        opt_lower=True,
        vocab_set=vocab,
    )
    assert out == b""  # per your rule, drop row if all <UNK>

def test_outbuf_reuse_produces_identical_output():
    # Provide an outbuf and ensure multiple calls reuse/clear it correctly
    outbuf = bytearray()
    a = process_tokens(b"Dog_NOUN", opt_lower=True, outbuf=outbuf)
    b = process_tokens(b"Cat_NOUN", opt_lower=True, outbuf=outbuf)
    assert a == b"dog"
    assert b == b"cat"
    # Ensure outbuf doesn't accumulate old content
    assert b != b"dog cat"
