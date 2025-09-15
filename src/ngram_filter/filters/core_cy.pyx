# ngram_filter/filters/core_cy.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=False
cimport cython
from cpython.bytes cimport PyBytes_AsString, PyBytes_FromStringAndSize

# ---- constants (bytes) ----
METADATA_PREFIX = b"__"
# Packed records: <year:uint64><match:uint64><volume:uint64> (little-endian)
FMT = "<QQQ"
SENTINEL_B = b"<UNK>"

# Google Books Ngram tag vocabulary (for recognition/stripping)
cdef set TAGS_VALID_G = {
    b"NOUN", b"PROPN", b"VERB", b"ADJ", b"ADV",
    b"PRON", b"DET", b"ADP", b"NUM", b"CONJ", b"X", b"."
}
# Mapping subset for lemmatizer (WordNet codes)
cdef dict MAP_G_TO_WN = {
    b"NOUN": "n",
    b"VERB": "v",
    b"ADJ":  "a",
    b"ADV":  "r",
}

# ======================== low-level: ASCII alpha check ========================

@cython.cfunc
@cython.inline
cdef bint _is_ascii_alpha_bytes(const unsigned char[:] buf):
    """
    Return 1 if buf is non-empty and consists only of ASCII letters A–Z/a–z.
    Return 0 otherwise.
    """
    cdef Py_ssize_t i, n = buf.shape[0]
    cdef unsigned char c
    if n == 0:
        return 0
    for i in range(n):
        c = buf[i]
        if c > 127:
            return 0
        if not (65 <= c <= 90 or 97 <= c <= 122):
            return 0
    return 1


# ======================== helpers ========================

@cython.cfunc
@cython.inline
cdef str _decode_token(bytes tok):
    # Always decode as UTF-8; alpha filtering is handled separately
    return tok.decode("utf-8", "surrogatepass")

@cython.cfunc
@cython.inline
cdef bytes _encode_token(str s, bint ascii_only):
    if ascii_only:
        try:
            return s.encode("ascii", "strict")
        except Exception:
            return SENTINEL_B
    else:
        return s.encode("utf-8", "surrogatepass")


# ======================== per-ngram processing (bytes-only) ========================

cpdef bytes process_tokens(
    bytes ngram,
    bint opt_lower = False,
    bint opt_alpha = False,
    bint opt_shorts = False,
    bint opt_stops = False,
    bint opt_lemmas = False,
    int  min_len = 3,
    object stop_set = None,
    object lemma_gen = None,
    object whitelist = None,
    bytearray outbuf = None
):
    """
    Bytes-only token pipeline:
      tokenize -> split POS (Google) -> lower -> whitelist -> alpha -> shorts -> stops -> lemmas
    Returns b"" if all tokens become <UNK>.
    """
    cdef Py_ssize_t N = ngram.__len__()
    if N == 0:
        return b""

    # flags
    cdef bint do_lower  = opt_lower
    cdef bint do_lemmas = (opt_lemmas and lemma_gen is not None)
    cdef bint do_whitelist = (whitelist is not None)
    cdef bint do_alpha  = opt_alpha
    cdef bint do_shorts = opt_shorts
    cdef bint do_stops  = (opt_stops and stop_set is not None)

    # prep output buffer (one-pass writer)
    if outbuf is None:
        outbuf = bytearray()
    else:
        outbuf.clear()

    # raw pointer for slicing without extra attribute lookups
    cdef char* base = PyBytes_AsString(ngram)

    cdef Py_ssize_t i = 0
    cdef Py_ssize_t tok_start, tok_end
    cdef Py_ssize_t token_count = 0
    cdef Py_ssize_t unk_count = 0

    cdef bytes tok_b, base_b, tag_b, out_token
    cdef Py_ssize_t last_uscore
    cdef bint is_unk

    # for lemma path
    cdef object wn_pos      # None or 'n'/'v'/'a'/'r'
    cdef str tok_s
    cdef str pos_s
    cdef str lem_s

    # scan tokens separated by ASCII space
    while i < N:
        # skip spaces
        while i < N and (<unsigned char>base[i]) == 32:
            i += 1
        if i >= N:
            break
        tok_start = i
        while i < N and (<unsigned char>base[i]) != 32:
            i += 1
        tok_end = i

        # materialize token bytes (needed for set lookups / .lower() / rfind)
        tok_b = PyBytes_FromStringAndSize(<char*>base + tok_start, tok_end - tok_start)

        # split POS by last underscore
        last_uscore = tok_b.rfind(b'_')  # find the last underscore
        wn_pos = None  # default to no POS
        if last_uscore > 0:  # check if there was a last underscore
            tag_b = tok_b[last_uscore + 1:]  # if there was, separate out the possible POS tag
            if tag_b in TAGS_VALID_G:  # see if what's after the last underscore is really a POS tag
                base_b = tok_b[:last_uscore]  # if it's a real POS tag, separate out the base token
                wn_pos = MAP_G_TO_WN.get(tag_b, None)  # and convert the tag to a WordNet code
            else:  # if what's after the last underscore *isn't* a POS tag...
                base_b = tok_b  # the base token is just the original token
        else:  # if there wasn't a last underscore...
            base_b = tok_b  # the base token is just the original token

        # Normalize token first (always needed)
        if do_lower:
            base_b = base_b.lower()
        if do_lemmas:
            tok_s = _decode_token(base_b)
            pos_s = wn_pos if wn_pos is not None else "n"
            res = lemma_gen.lemmatize(tok_s, pos=pos_s)
            normalized_token = _encode_token(<str> res, do_alpha)
        else:
            normalized_token = base_b

        # BRANCH: whitelist vs. normal filtering
        if do_whitelist:
            # Whitelist path: simple check, no other filters
            if normalized_token in whitelist:
                out_token = normalized_token
            else:
                out_token = SENTINEL_B
                unk_count += 1
        else:
            # Normal filtering path: apply all checks
            is_unk = 0
            if do_alpha and not _is_ascii_alpha_bytes(normalized_token):
                is_unk = 1
            elif do_shorts and normalized_token.__len__() < min_len:
                is_unk = 1
            elif do_stops and normalized_token in stop_set:
                is_unk = 1

            if is_unk:
                out_token = SENTINEL_B
                unk_count += 1
            else:
                out_token = normalized_token

        # write token (single exit path)
        if token_count > 0:
            outbuf.append(32)  # ' '
        outbuf.extend(out_token)
        token_count += 1

    # if everything became <UNK>, return empty
    if token_count > 0 and unk_count == token_count:
        return b""

    return bytes(outbuf)
