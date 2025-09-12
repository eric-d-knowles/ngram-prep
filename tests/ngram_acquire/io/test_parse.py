# tests/io/test_parse.py
from ngram_acquire.io.parse import parse_line
from ngram_acquire.utils.filters import make_ngram_type_predicate


def test_empty_or_whitespace_returns_none():
    assert parse_line("") == (None, None)
    assert parse_line("   ") == (None, None)


def test_no_tab_separator_returns_none():
    assert parse_line("cat_NOUN") == (None, None)


def test_parses_single_tuple():
    line = "cat_NOUN\t1990,12,3"
    key, rec = parse_line(line)
    assert key == "cat_NOUN"
    assert rec == {
        "frequencies": [{"year": 1990, "frequency": 12, "document_count": 3}]
    }


def test_parses_multiple_tuples_and_order_is_preserved():
    line = "cat_NOUN dog_VERB\t1990,1,1\t1991,2,2\t1992,3,3"
    key, rec = parse_line(line)
    assert key == "cat_NOUN dog_VERB"
    ys = [f["year"] for f in rec["frequencies"]]
    assert ys == [1990, 1991, 1992]


def test_skips_malformed_tuples_but_keeps_valid():
    line = "x_NOUN\tBAD\t1999,10,2\tnope,,-1\t2000,5,1"
    key, rec = parse_line(line)
    assert key == "x_NOUN"
    pairs = [(f["year"], f["frequency"]) for f in rec["frequencies"]]
    assert pairs == [(1999, 10), (2000, 5)]


def test_requires_at_least_one_valid_tuple():
    line = "x_NOUN\tBAD\talsobad"
    assert parse_line(line) == (None, None)


def test_predicate_filters_before_parsing_tagged_only():
    pred_tagged = make_ngram_type_predicate("tagged")
    # no tags -> filtered out
    assert parse_line("alpha beta\t1990,1,1", filter_pred=pred_tagged) \
        == (None, None)
    # all tagged -> kept
    key, rec = parse_line(
        "alpha_NOUN beta_VERB\t1990,1,1\t1991,2,2", filter_pred=pred_tagged
    )
    assert key == "alpha_NOUN beta_VERB"
    assert len(rec["frequencies"]) == 2


def test_predicate_untagged_only_rejects_mixed_lines():
    pred_untagged = make_ngram_type_predicate("untagged")
    # mixed -> reject
    assert parse_line("x_NOUN y\t1990,1,1", filter_pred=pred_untagged) \
        == (None, None)
    # all untagged -> keep
    key, rec = parse_line("a b\t1990,1,1\t1991,2,2",
                                filter_pred=pred_untagged)
    assert key == "a b"
    assert [f["year"] for f in rec["frequencies"]] == [1990, 1991]
