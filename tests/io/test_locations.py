# tests/io/test_locations.py
import re
import pytest

from ngram_prep.io.locations import set_location_info, BASE_URL


def test_builds_correct_url_and_regex():
    url, rx = set_location_info(2, "20200217", "eng-us")
    assert url == (
        f"{BASE_URL}/20200217/eng-us/eng-us-2-ngrams_exports.html"
    )
    assert isinstance(rx, re.Pattern)
    assert rx.fullmatch("eng-us-2-00001-of-00024.gz")
    assert not rx.fullmatch("2-00001-of-00024.gz")         # missing corpus
    assert not rx.fullmatch("eng-us-3-00001-of-00024.gz")  # wrong n


def test_validates_inputs():
    with pytest.raises(ValueError):
        set_location_info(0, "20200217", "eng")           # out of range
    with pytest.raises(ValueError):
        set_location_info(6, "20200217", "eng")
    with pytest.raises(ValueError):
        set_location_info(1, "2020-02-17", "eng")         # not 8 digits
    with pytest.raises(ValueError):
        set_location_info(1, "20200217", "eng_us")        # invalid char
