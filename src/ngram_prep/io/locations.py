# ngram_prep/io/locations.py
from __future__ import annotations

import re
from typing import Tuple

BASE_URL = "https://storage.googleapis.com/books/ngrams/books"


def set_location_info(
    ngram_size: int,
    repo_release_id: str,
    repo_corpus_id: str,
) -> Tuple[str, re.Pattern[str]]:
    """
    Build the export page URL and a compiled regex for the file names.

    Examples
    --------
    set_location_info(2, "20200217", "eng-us")
      -> (
           ".../books/20200217/eng-us/eng-us-2-ngrams_exports.html",
           re.compile(r"^eng-us-2-\\d{5}-of-\\d{5}\\.gz$")
         )
    """
    if ngram_size not in (1, 2, 3, 4, 5):
        raise ValueError("ngram_size must be an integer in 1..5")

    # Release IDs are 8-digit dates like 20200217
    if not re.fullmatch(r"\d{8}", repo_release_id):
        raise ValueError("repo_release_id must be an 8-digit YYYYMMDD string")

    # Corpus IDs are like "eng", "eng-us", "eng-fiction" etc.
    if not re.fullmatch(r"[A-Za-z0-9-]+", repo_corpus_id):
        raise ValueError("repo_corpus_id must contain only [A-Za-z0-9-]")

    page_url = (
        f"{BASE_URL}/{repo_release_id}/{repo_corpus_id}/"
        f"{repo_corpus_id}-{ngram_size}-ngrams_exports.html"
    )

    # Files look like: "<corpus>-<n>-00000-of-00024.gz"
    file_rx = re.compile(
        rf"^{re.escape(repo_corpus_id)}-{ngram_size}-\d{{5}}-of-\d{{5}}\.gz$"
    )

    return page_url, file_rx
