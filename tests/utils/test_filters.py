# tests/test_ngram_type_predicate.py
import pytest

from ngram_prep.utils.filters import make_ngram_type_predicate


def test_tagged_accepts_all_tokens_with_valid_suffix():
    pred = make_ngram_type_predicate("tagged")
    assert pred("cat_NOUN dog_VERB")
    assert pred("__NOUN")
    assert not pred("cat dog")
    assert not pred("cat_NOUN dog")      # mixed: one token lacks a tag
    assert not pred("_NOUN_")            # tag not at token end


def test_untagged_accepts_only_plain_tokens():
    pred = make_ngram_type_predicate("untagged")
    assert pred("alpha beta")
    assert not pred("alpha_NOUN beta")
    assert not pred("__NOUN")            # this is considered tagged by rule


def test_all_predicate_nonempty_only():
    pred = make_ngram_type_predicate("all")
    assert pred("a b")
    assert pred("c_NOUN")
    assert not pred("")
    assert not pred("   ")


def test_unknown_tag_treated_as_untagged():
    pred_tagged = make_ngram_type_predicate("tagged")
    pred_untagged = make_ngram_type_predicate("untagged")
    assert not pred_tagged("cat_XYZ")
    assert pred_untagged("cat_XYZ")


def test_custom_tagset_supported():
    pred = make_ngram_type_predicate("tagged", tags=("ZZZ",))
    assert pred("foo_ZZZ")
    assert not pred("foo_NOUN")


def test_mixed_tagged_and_untagged_rejected_by_tagged_and_untagged():
    pred_tagged = make_ngram_type_predicate("tagged")
    pred_untagged = make_ngram_type_predicate("untagged")
    line = "x_NOUN y"
    assert not pred_tagged(line)
    assert not pred_untagged(line)


def test_invalid_ngram_type_raises():
    with pytest.raises(ValueError):
        make_ngram_type_predicate("invalid")
