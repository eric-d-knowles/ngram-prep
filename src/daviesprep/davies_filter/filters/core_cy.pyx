# daviesprep/davies_filter/filters/core_cy.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False, embedsignature=False
cimport cython
from cpython.bytes cimport PyBytes_AsString, PyBytes_FromStringAndSize

# ---- constants ----
SENTINEL_B = b"<UNK>"

# ======================== low-level: alphabetic check ========================

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


@cython.cfunc
@cython.inline
cdef bint _is_unicode_alpha(str s):
    """
    Return 1 if string is non-empty and consists only of Unicode alphabetic characters.
    Return 0 otherwise.
    """
    if not s:
        return 0
    return s.isalpha()


# ======================== helpers ========================

@cython.cfunc
@cython.inline
cdef str _decode_token(bytes tok):
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


# ======================== sentence processing ========================

cpdef bytes process_sentence(
    bytes sentence,
    bint opt_lower = False,
    bint opt_alpha = False,
    bint opt_ascii_alpha_only = False,
    bint opt_shorts = False,
    bint opt_stops = False,
    bint opt_lemmas = False,
    int  min_len = 3,
    object stop_set = None,
    object lemma_gen = None,
    object whitelist = None,
    object always_include = None,
    bytearray outbuf = None
):
    """
    Process a sentence (space-separated tokens):
      tokenize -> lower -> whitelist (if enabled) -> alpha -> shorts -> stops -> lemmas

    Returns b"" if all tokens become <UNK> or fewer than 2 tokens remain.

    Unlike ngram processing, Davies sentences have no POS tags embedded.

    If whitelist is provided, tokens not in whitelist become <UNK> (checked after lowercasing).
    If always_include is provided, those tokens are preserved even if not in whitelist.
    
    When opt_alpha=True:
      - If opt_ascii_alpha_only=True: reject all non-ASCII characters, keep only A-Z/a-z
      - If opt_ascii_alpha_only=False: accept Unicode alphabetic characters from any language
    """
    cdef Py_ssize_t N = sentence.__len__()
    if N == 0:
        return b""

    # flags
    cdef bint do_lower  = opt_lower
    cdef bint do_lemmas = (opt_lemmas and lemma_gen is not None)
    cdef bint do_alpha  = opt_alpha
    cdef bint do_shorts = opt_shorts
    cdef bint do_stops  = (opt_stops and stop_set is not None)
    cdef bint do_whitelist = (whitelist is not None)
    cdef bint do_always_include = (always_include is not None)

    # prep output buffer
    if outbuf is None:
        outbuf = bytearray()
    else:
        outbuf.clear()

    # raw pointer for slicing
    cdef char* base = PyBytes_AsString(sentence)

    cdef Py_ssize_t i = 0
    cdef Py_ssize_t tok_start, tok_end
    cdef Py_ssize_t token_count = 0
    cdef Py_ssize_t unk_count = 0

    cdef bytes tok_b, out_token
    cdef bint is_unk

    # for lemma path
    cdef str tok_s
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

        # materialize token bytes
        tok_b = PyBytes_FromStringAndSize(<char*>base + tok_start, tok_end - tok_start)

        # Apply filters
        is_unk = 0

        # Lowercase first
        if do_lower:
            tok_b = tok_b.lower()

        # Whitelist filter (after lowercasing, before other filters)
        if not is_unk and do_whitelist:
            if tok_b not in whitelist:
                # Check if it's in always_include before marking as UNK
                if not do_always_include or tok_b not in always_include:
                    is_unk = 1

        # Alpha filter (skip for always_include tokens)
        if do_alpha:
            # Skip alpha check if token is in always_include set
            if not (do_always_include and tok_b in always_include):
                if not _is_ascii_alpha_bytes(tok_b):
                    # Contains non-ASCII bytes
                    if opt_ascii_alpha_only:
                        # If ascii_alpha_only=True, reject any non-ASCII bytes
                        # (even if they're valid Unicode alpha chars)
                        is_unk = 1
                    else:
                        # Otherwise, accept all Unicode alphabetic characters
                        # (including accented letters, Greek, Cyrillic, Chinese, etc.)
                        try:
                            tok_s = _decode_token(tok_b)
                            if not _is_unicode_alpha(tok_s):
                                is_unk = 1
                        except:
                            # Decoding failed, mark as invalid
                            is_unk = 1

        # Length filter (skip for always_include tokens)
        if not is_unk and do_shorts:
            if not (do_always_include and tok_b in always_include):
                if tok_b.__len__() < min_len:
                    is_unk = 1

        # Stopword filter (skip for always_include tokens)
        if not is_unk and do_stops:
            if not (do_always_include and tok_b in always_include):
                if tok_b in stop_set:
                    is_unk = 1

        # Lemmatization (after all filters pass, but skip for always_include tokens)
        if not is_unk and do_lemmas:
            if do_always_include and tok_b in always_include:
                pass  # Token is protected, skip lemmatization
            else:
                tok_s = _decode_token(tok_b)
                # Default to NOUN for Davies data (no POS tags)
                res = lemma_gen.lemmatize(tok_s, pos="NOUN")
                tok_b = _encode_token(<str> res, opt_ascii_alpha_only)

        # Write token
        if is_unk:
            out_token = SENTINEL_B
            unk_count += 1
        else:
            out_token = tok_b

        if token_count > 0:
            outbuf.append(32)  # ' '
        outbuf.extend(out_token)
        token_count += 1

    # Reject if all tokens are UNK or fewer than 2 tokens remain
    cdef Py_ssize_t valid_tokens = token_count - unk_count
    if valid_tokens < 2:
        return b""

    return bytes(outbuf)
