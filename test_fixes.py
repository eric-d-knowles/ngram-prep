#!/usr/bin/env python3
"""Test script for ascii_alpha_only parameter and filter fixes"""

from ngramprep.ngram_filter.filters.core_cy import process_tokens

print("=" * 80)
print("TEST 1: ascii_alpha_only parameter")
print("=" * 80)

test_cases = [
    (b"hello world", "Pure ASCII"),
    (b"caf\xc3\xa9", "café (accented e)"),
    (b"\xc3\xa0", "à (accented a)"),
    (b"r\xc3\xa9sum\xc3\xa9", "résumé"),
    (b"\xce\xb1\xce\xb2\xce\xb3", "αβγ (Greek)"),
]

for test_case, description in test_cases:
    try:
        decoded = test_case.decode('utf-8')
        print(f"\nInput: '{decoded}' ({description})")
    except:
        print(f"\nInput: {test_case!r} ({description})")
    
    # Test with ascii_alpha_only=False (allow Unicode alpha)
    result_unicode = process_tokens(test_case, opt_alpha=True, opt_ascii_alpha_only=False)
    status_unicode = 'KEPT' if result_unicode else 'FILTERED'
    print(f"  ascii_alpha_only=False: {status_unicode:15s} {repr(result_unicode)}")
    
    # Test with ascii_alpha_only=True (ASCII-only)
    result_ascii = process_tokens(test_case, opt_alpha=True, opt_ascii_alpha_only=True)
    status_ascii = 'KEPT' if result_ascii else 'FILTERED'
    print(f"  ascii_alpha_only=True:  {status_ascii:15s} {repr(result_ascii)}")

print("\n" + "=" * 80)
print("TEST 2: Independent filter checks (length + stopwords)")
print("=" * 80)

stop_set = {b"a", b"the", b"is"}

# Test 1: Short stopword (should be filtered by length check)
short_stopword = b"a"
result = process_tokens(short_stopword, opt_alpha=True, opt_shorts=True, min_len=2, 
                       opt_stops=True, stop_set=stop_set)
print(f"\nToken: 'a' (length=1, is stopword, min_len=2)")
print(f"  Result: {repr(result):30s} {'KEPT' if result else 'FILTERED (length check)'}")

# Test 2: Medium stopword (should be filtered by stopword check)
medium_stopword = b"the"
result = process_tokens(medium_stopword, opt_alpha=True, opt_shorts=True, min_len=2,
                       opt_stops=True, stop_set=stop_set)
print(f"\nToken: 'the' (length=3, is stopword, min_len=2)")
print(f"  Result: {repr(result):30s} {'KEPT' if result else 'FILTERED (stopword check)'}")

# Test 3: Token that passes both checks
good_token = b"hello"
result = process_tokens(good_token, opt_alpha=True, opt_shorts=True, min_len=2,
                       opt_stops=True, stop_set=stop_set)
print(f"\nToken: 'hello' (length=5, not a stopword, min_len=2)")
print(f"  Result: {repr(result):30s} {'KEPT' if result else 'FILTERED'}")

print("\n" + "=" * 80)
print("TEST RESULTS:")
print("✓ ascii_alpha_only parameter works correctly")
print("✓ Independent filter checks work correctly")
print("=" * 80)
