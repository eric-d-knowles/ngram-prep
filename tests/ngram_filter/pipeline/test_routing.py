# tests/test_routing.py
from typing import Optional

import pytest

from ngram_filter.pipeline.old.routing import Router, byte_ranges


# ---------------------------- Helpers ----------------------------

def _prefix_to_int(b: bytes) -> int:
    return int.from_bytes(b, "big")


def _range_size(lo: bytes, hi: Optional[bytes], prefix_len: int) -> int:
    lo_i = _prefix_to_int(lo)
    hi_i = 256 ** prefix_len if hi is None else _prefix_to_int(hi)
    return hi_i - lo_i


def _ranges_to_ints(ranges, prefix_len: int):
    """Convert (lo, hi) byte ranges to (lo_i, hi_i) integer pairs for checks."""
    out = []
    for lo, hi in ranges:
        lo_i = _prefix_to_int(lo)
        hi_i = 256 ** prefix_len if hi is None else _prefix_to_int(hi)
        out.append((lo_i, hi_i))
    return out


# ---------------------------- Router tests ----------------------------

@pytest.mark.parametrize(
    "num_shards, inner_lanes",
    [
        (1, 1),
        (1, 16),   # power of two
        (4, 1),
        (4, 16),   # power of two
        (7, 10),   # non-power-of-two lanes
    ],
)
def test_router_ranges_and_stability(num_shards, inner_lanes):
    r = Router(num_shards=num_shards, inner_lanes=inner_lanes, seed=123)
    keys = [f"k{i}".encode() for i in range(100)]
    outs = [r.route(k) for k in keys]

    # Bounds
    for outer, inner in outs:
        assert 0 <= outer < num_shards
        assert 0 <= inner < inner_lanes

    # Deterministic with same seed
    assert [r.route(k) for k in keys] == outs

    # Seed sensitivity only makes sense if there’s >1 bucket total
    if num_shards > 1 or inner_lanes > 1:
        r2 = Router(num_shards=num_shards, inner_lanes=inner_lanes, seed=999)
        outs2 = [r2.route(k) for k in keys]
        assert any(a != b for a, b in zip(outs, outs2))


def test_router_single_lane_is_zero():
    r = Router(num_shards=8, inner_lanes=1)
    for i in range(50):
        outer, inner = r.route(f"key{i}".encode())
        assert 0 <= outer < 8
        assert inner == 0


def test_router_single_shard_is_zero():
    r = Router(num_shards=1, inner_lanes=8)
    for i in range(50):
        outer, inner = r.route(f"key{i}".encode())
        assert outer == 0
        assert 0 <= inner < 8


def test_router_rejects_bad_args():
    with pytest.raises(ValueError):
        Router(num_shards=0)
    with pytest.raises(ValueError):
        Router(num_shards=4, inner_lanes=0)


# ---------------------------- byte_ranges tests ----------------------------

@pytest.mark.parametrize("prefix_len", [1, 2])
@pytest.mark.parametrize("num_ranges", [1, 2, 3, 4, 7, 16, 31])
def test_byte_ranges_cover_exactly_and_non_overlapping(num_ranges, prefix_len):
    total = 256 ** prefix_len
    ranges = byte_ranges(num_ranges=num_ranges, prefix_len=prefix_len)

    # Basic shape: last hi is None
    assert len(ranges) == num_ranges
    assert ranges[-1][1] is None

    # Convert to integer intervals and check contiguity & no overlap
    int_ranges = _ranges_to_ints(ranges, prefix_len)
    # They should be sorted and contiguous
    assert int_ranges[0][0] == 0
    for (a_lo, a_hi), (b_lo, b_hi) in zip(int_ranges, int_ranges[1:]):
        assert a_hi == b_lo  # contiguous
        assert a_lo < a_hi
        assert b_lo < b_hi

    # Cover exactly the domain
    assert int_ranges[-1][1] == total

    # Widths differ by at most 1
    widths = [hi - lo for lo, hi in int_ranges]
    assert max(widths) - min(widths) <= 1


def test_byte_ranges_raise_on_bad_args():
    with pytest.raises(ValueError):
        byte_ranges(0, prefix_len=1)
    with pytest.raises(ValueError):
        byte_ranges(300, prefix_len=1)  # > 256**1
    with pytest.raises(ValueError):
        byte_ranges(1, prefix_len=0)


def test_byte_ranges_example_small():
    rs = byte_ranges(4, prefix_len=1)
    # Explicit sizes should sum to 256
    sizes = [_range_size(lo, hi, prefix_len=1) for lo, hi in rs]
    assert sum(sizes) == 256
    # First lo must be b'\\x00', last hi must be None
    assert rs[0][0] == b"\x00"
    assert rs[-1][1] is None


def test_byte_ranges_prefix_len_2_more_slices():
    # Pick a weird slice count to exercise remainder distribution
    rs = byte_ranges(37, prefix_len=2)
    sizes = [_range_size(lo, hi, prefix_len=2) for lo, hi in rs]
    assert sum(sizes) == 256 ** 2
    assert max(sizes) - min(sizes) <= 1
    # Ensure strictly increasing lows
    lows = [r[0] for r in rs]
    assert lows == sorted(lows)


# ---------------------------- Distribution smoke test ----------------------------

def test_router_reasonable_spread_power_of_two_lanes():
    """Not a statistical test—just checks distribution isn't pathological."""
    r = Router(num_shards=8, inner_lanes=16, seed=42)
    keys = [f"k{i}".encode() for i in range(2000)]
    shard_counts = [0] * r.num_shards
    for k in keys:
        shard_counts[r.route_outer(k)] += 1

    # Expect roughly uniform distribution; allow wide tolerance to avoid flakiness
    mean = sum(shard_counts) / len(shard_counts)
    for c in shard_counts:
        assert abs(c - mean) <= 0.20 * mean  # 20% tolerance


def test_router_reasonable_spread_non_power_of_two_lanes():
    r = Router(num_shards=7, inner_lanes=10, seed=99)
    keys = [f"x{i}".encode() for i in range(2100)]
    lane_counts = [0] * r.inner_lanes
    for k in keys:
        lane_counts[r.route_inner(k)] += 1

    mean = sum(lane_counts) / len(lane_counts)
    for c in lane_counts:
        assert abs(c - mean) <= 0.25 * mean  # slightly looser tolerance
