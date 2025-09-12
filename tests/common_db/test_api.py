# tests/test_api.py
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from common_db.api import open_db, prefix_scan, range_scan


# =============================================================================
# Open/close, profiles, bulk finalize, path types, and robustness (original)
# =============================================================================

class TestOpenDb:
    """Test the open_db context manager"""

    def test_open_db_defaults(self):
        """Test open_db with default parameters"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with open_db(db_path) as db:
                # Should open in read-write mode by default
                assert db is not None
                # Basic operation to verify it works
                db.put(b"test_key", b"test_value")
                assert db.get(b"test_key") == b"test_value"

    def test_open_db_read_only(self):
        """Test open_db with read-only mode"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # First create a database
            with open_db(db_path, mode="rw") as db:
                db.put(b"existing_key", b"existing_value")

            # Then open read-only
            with open_db(db_path, mode="ro") as db:
                assert db.get(b"existing_key") == b"existing_value"
                # Write attempts may be disallowed by shim (implementation dependent)

    def test_open_db_with_profile(self):
        """Test open_db with different profiles"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            for profile in ["read", "write", "bulk"]:
                with open_db(db_path, profile=profile) as db:
                    assert db is not None
                    # Basic sanity check
                    db.put(f"profile_{profile}".encode(), b"test_value")
                    assert db.get(f"profile_{profile}".encode()) == b"test_value"

    def test_open_db_bulk_finalization(self):
        """Test bulk profile with finalization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with open_db(db_path, profile="bulk", finalize_bulk_to="read") as db:
                # Add some bulk data
                for i in range(10):
                    db.put(f"bulk_key_{i}".encode(), f"bulk_value_{i}".encode())

                # Verify data is there
                assert db.get(b"bulk_key_5") == b"bulk_value_5"

            # After context exit, finalization should have run
            # Verify we can still read the data
            with open_db(db_path, mode="ro") as db:
                assert db.get(b"bulk_key_5") == b"bulk_value_5"

    def test_open_db_path_types(self):
        """Test open_db accepts both string and Path objects"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with string path
            db_path_str = str(Path(tmpdir) / "test_str.db")
            with open_db(db_path_str) as db:
                db.put(b"str_key", b"str_value")
                assert db.get(b"str_key") == b"str_value"

            # Test with Path object
            db_path_obj = Path(tmpdir) / "test_path.db"
            with open_db(db_path_obj) as db:
                db.put(b"path_key", b"path_value")
                assert db.get(b"path_key") == b"path_value"

    @patch('common_db.api.rs.DB.open')
    def test_open_db_exception_handling(self, mock_db_open):
        """Test that exceptions during finalization don't crash"""
        # Create a mock DB that raises exceptions on finalization methods
        mock_db = Mock()
        mock_db.finalize_bulk.side_effect = Exception("Finalize failed")
        mock_db.compact_all.side_effect = Exception("Compact failed")
        mock_db.set_profile.side_effect = Exception("Set profile failed")
        mock_db.close.side_effect = Exception("Close failed")
        mock_db_open.return_value = mock_db

        # Should not raise exceptions despite failures in finalization
        with open_db("/fake/path", profile="bulk"):
            pass  # Should complete without raising

        # Verify all methods were attempted
        mock_db.finalize_bulk.assert_called_once()
        mock_db.set_profile.assert_called()
        mock_db.close.assert_called_once()

    @patch('common_db.api.rs.DB.open')
    def test_open_db_no_finalization_methods(self, mock_db_open):
        """Test graceful handling when DB doesn't have finalization methods"""
        # Create a mock DB without finalization methods
        mock_db = Mock()
        if hasattr(mock_db, "finalize_bulk"):
            del mock_db.finalize_bulk
        if hasattr(mock_db, "compact_all"):
            del mock_db.compact_all
        if hasattr(mock_db, "set_profile"):
            del mock_db.set_profile
        if hasattr(mock_db, "close"):
            del mock_db.close
        mock_db_open.return_value = mock_db

        # Should work fine even without these methods
        with open_db("/fake/path", profile="bulk"):
            pass


# =============================================================================
# Prefix scan basic behavior (original)
# =============================================================================

class TestPrefixScan:
    """Test the prefix_scan function"""

    def test_prefix_scan_basic(self):
        """Test basic prefix scanning functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with open_db(db_path) as db:
                # Insert test data
                test_data = [
                    (b"prefix:key1", b"value1"),
                    (b"prefix:key2", b"value2"),
                    (b"prefix:key3", b"value3"),
                    (b"other:key1", b"other_value1"),
                    (b"another:key", b"another_value"),
                ]

                for key, value in test_data:
                    db.put(key, value)

                # Scan with prefix
                results = list(prefix_scan(db, b"prefix:"))

                # Should get exactly 3 results with prefix
                assert len(results) == 3
                expected_keys = {b"prefix:key1", b"prefix:key2", b"prefix:key3"}
                found_keys = {key for key, value in results}
                assert found_keys == expected_keys

                # Verify values are correct
                result_dict = dict(results)
                assert result_dict[b"prefix:key1"] == b"value1"
                assert result_dict[b"prefix:key2"] == b"value2"
                assert result_dict[b"prefix:key3"] == b"value3"

    def test_prefix_scan_empty_result(self):
        """Test prefix scan with no matching keys"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with open_db(db_path) as db:
                db.put(b"some_key", b"some_value")

                # Scan for non-existent prefix
                results = list(prefix_scan(db, b"nonexistent:"))
                assert len(results) == 0

    def test_prefix_scan_partial_match(self):
        """Test that prefix scan doesn't include partial matches"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with open_db(db_path) as db:
                # Insert keys where one is a prefix of another
                db.put(b"test", b"value1")
                db.put(b"test:", b"value2")
                db.put(b"test:key", b"value3")

                # Scan for "test:" should only get the exact matches
                results = list(prefix_scan(db, b"test:"))
                result_keys = [key for key, value in results]

                assert b"test:" in result_keys
                assert b"test:key" in result_keys
                assert b"test" not in result_keys  # Should not include partial match


# =============================================================================
# Range scan basic behavior (original)
# =============================================================================

class TestRangeScan:
    """Test the range_scan function"""

    def test_range_scan_basic(self):
        """Test basic range scanning functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with open_db(db_path) as db:
                # Insert test data (keys will be sorted lexicographically)
                test_data = [
                    (b"a", b"value_a"),
                    (b"b", b"value_b"),
                    (b"c", b"value_c"),
                    (b"d", b"value_d"),
                    (b"e", b"value_e"),
                ]

                for key, value in test_data:
                    db.put(key, value)

                # Range scan from 'b' to 'd' (exclusive)
                results = list(range_scan(db, b"b", b"d"))

                # Should get 'b' and 'c', but not 'd'
                assert len(results) == 2
                result_keys = [key for key, value in results]
                assert b"b" in result_keys
                assert b"c" in result_keys
                assert b"d" not in result_keys

                # Verify values
                result_dict = dict(results)
                assert result_dict[b"b"] == b"value_b"
                assert result_dict[b"c"] == b"value_c"

    def test_range_scan_empty_range(self):
        """Test range scan with empty range"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with open_db(db_path) as db:
                db.put(b"a", b"value_a")
                db.put(b"z", b"value_z")

                # Range that doesn't include any keys
                results = list(range_scan(db, b"m", b"n"))
                assert len(results) == 0

    def test_range_scan_single_key(self):
        """Test range scan that should return single key"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with open_db(db_path) as db:
                db.put(b"key1", b"value1")
                db.put(b"key3", b"value3")

                # Range that includes only key1
                results = list(range_scan(db, b"key1", b"key2"))
                assert len(results) == 1
                assert results[0] == (b"key1", b"value1")

    def test_range_scan_boundary_conditions(self):
        """Test range scan boundary conditions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with open_db(db_path) as db:
                # Test with identical lower and upper bounds
                db.put(b"test", b"value")
                results = list(range_scan(db, b"test", b"test"))
                assert len(results) == 0  # Upper is exclusive

                # Test with lower > upper
                results = list(range_scan(db, b"z", b"a"))
                assert len(results) == 0


# =============================================================================
# Integration (original)
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_scan_functions_with_bulk_profile(self):
        """Test scan functions work correctly with bulk profile"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Use bulk profile to load data
            with open_db(db_path, profile="bulk") as db:
                # Load bulk data
                for i in range(100):
                    key = f"bulk:{i:03d}".encode()
                    value = f"value_{i}".encode()
                    db.put(key, value)

                for i in range(50):
                    key = f"other:{i:03d}".encode()
                    value = f"other_value_{i}".encode()
                    db.put(key, value)

            # Now scan the data
            with open_db(db_path, mode="ro", profile="read") as db:
                # Test prefix scan
                bulk_results = list(prefix_scan(db, b"bulk:"))
                assert len(bulk_results) == 100

                # Test range scan
                range_results = list(range_scan(db, b"bulk:020", b"bulk:030"))
                assert len(range_results) == 10  # 020-029

                # Verify some specific values
                result_dict = dict(bulk_results)
                assert result_dict[b"bulk:050"] == b"value_50"


# =====================================================================
# Fixtures (original)
# =====================================================================

@pytest.fixture
def temp_db_path():
    """Fixture providing a temporary database path"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def populated_db(temp_db_path):
    """Fixture providing a pre-populated database"""
    with open_db(temp_db_path) as db:
        # Add some standard test data
        test_data = [
            (b"user:alice", b"{'name': 'Alice', 'age': 30}"),
            (b"user:bob", b"{'name': 'Bob', 'age': 25}"),
            (b"user:charlie", b"{'name': 'Charlie', 'age': 35}"),
            (b"config:timeout", b"300"),
            (b"config:retries", b"3"),
            (b"stats:2024-01", b"{'requests': 1000}"),
            (b"stats:2024-02", b"{'requests': 1500}"),
        ]
        for key, value in test_data:
            db.put(key, value)
    yield temp_db_path


class TestWithFixtures:
    """Tests using pytest fixtures"""

    def test_prefix_scan_users(self, populated_db):
        """Test prefix scanning for users using fixture"""
        with open_db(populated_db, mode="ro") as db:
            user_results = list(prefix_scan(db, b"user:"))
            assert len(user_results) == 3

            user_names = [key.split(b":")[1] for key, value in user_results]
            assert b"alice" in user_names
            assert b"bob" in user_names
            assert b"charlie" in user_names

    def test_range_scan_stats(self, populated_db):
        """Test range scanning for stats using fixture"""
        with open_db(populated_db, mode="ro") as db:
            # ';' comes after ':' in ASCII, so [b"stats:", b"stats;") bounds the "stats:" prefix range
            stats_results = list(range_scan(db, b"stats:", b"stats;"))
            assert len(stats_results) == 2

            result_dict = dict(stats_results)
            assert b"stats:2024-01" in result_dict
            assert b"stats:2024-02" in result_dict


# =============================================================================
# Additional edge cases & call-shape checks (new)
# =============================================================================

def test_prefix_scan_upper_bound_boundary():
    """Keys that start with the prefix are included; immediate next-lex keys are excluded."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.db"
        with open_db(p) as db:
            for k in [b"ab:", b"ab:0", b"ab:9", b"ab;0", b"ab", b"ac:0"]:
                db.put(k, b"v")

        with open_db(p, mode="ro") as db:
            ks = [k for k, _ in prefix_scan(db, b"ab:")]
            assert set(ks) == {b"ab:", b"ab:0", b"ab:9"}
            assert b"ab;0" not in ks  # next-lex after ":" should not appear
            assert b"ab" not in ks    # not a prefix match
            assert b"ac:0" not in ks  # different prefix


def test_prefix_scan_binary_safe_keys():
    """Prefix scan should be purely byte-wise, not UTF-8 sensitive."""
    import tempfile
    from pathlib import Path
    from common_db.api import open_db, prefix_scan

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dbdir = root / "dbdir"
        dbdir.mkdir(parents=True, exist_ok=True)  # ensure parent dir exists
        p = dbdir / "b.db"

        with open_db(p) as db:
            db.put(b"\x00bin:\xff", b"\x00\xff")
            db.put(b"\x00bin:\x01", b"\x01")
            db.put(b"\x01bin:\x00", b"nope")

        # reopen (RO or RW are both fine; use RO to mirror other tests)
        with open_db(p, mode="ro") as db:
            out = dict(prefix_scan(db, b"\x00bin:"))
            assert out[b"\x00bin:\xff"] == b"\x00\xff"
            assert out[b"\x00bin:\x01"] == b"\x01"
            # Only the chosen binary prefix should match
            assert b"\x01bin:\x00" not in out


@pytest.mark.parametrize(
    "lower,upper,expected",
    [
        (b"b", b"d", [b"b", b"c"]),
        (b"a", b"a", []),                       # empty (upper is exclusive)
        (b"a", b"e", [b"a", b"b", b"c", b"d"]), # full span
    ],
)
def test_range_scan_param(lower, upper, expected, tmp_path: Path):
    p = tmp_path / "r.db"
    with open_db(p) as db:
        for k in [b"a", b"b", b"c", b"d"]:
            db.put(k, b"v")

    with open_db(p, mode="ro") as db:
        got = [k for k, _ in range_scan(db, lower, upper)]
        assert got == expected


def test_range_scan_reversed_is_empty(tmp_path: Path):
    p = tmp_path / "rev.db"
    with open_db(p) as db:
        for k in [b"a", b"b", b"c"]:
            db.put(k, b"v")

    with open_db(p, mode="ro") as db:
        assert list(range_scan(db, b"z", b"a")) == []


def test_range_scan_binary_bounds(tmp_path: Path):
    """Binary keys with explicit upper-exclusion bound."""
    p = tmp_path / "bin.db"
    with open_db(p) as db:
        keys = [b"\x00", b"\x00\x01", b"\x00\x01\xff", b"\x00\x02", b"\x01"]
        for k in keys:
            db.put(k, b"v")

    with open_db(p, mode="ro") as db:
        # Expect all keys k such that b"\x00\x01" <= k < b"\x00\x02"
        got = [k for k, _ in range_scan(db, b"\x00\x01", b"\x00\x02")]
        assert got == [b"\x00\x01", b"\x00\x01\xff"]


@patch("common_db.api.rs.DB.open")
def test_open_invocation_shape_read_only(mock_open):
    """Ensure api passes the expected flags to the shim for RO opens."""
    # Simulate returning a minimal object that can be used in a context
    class Dummy:
        def close(self): pass
        def iterator(self):
            class It:
                def seek(self, *_): pass
                def valid(self): return False
                def next(self): pass
                def key(self): return b""
                def value(self): return b""
            return It()
    mock_open.return_value = Dummy()

    with open_db("/fake/path", mode="ro"):
        pass

    mock_open.assert_called_with("/fake/path", read_only=True, create_if_missing=False)


@patch("common_db.api.rs.DB.open")
def test_open_invocation_shape_read_write(mock_open):
    """Ensure api passes the expected flags to the shim for RW opens."""
    class Dummy:
        def close(self): pass
        def put(self, *_): pass
        def get(self, *_): return None
        def iterator(self):
            class It:
                def seek(self, *_): pass
                def valid(self): return False
                def next(self): pass
                def key(self): return b""
                def value(self): return b""
            return It()
    mock_open.return_value = Dummy()

    with open_db("/fake/path", mode="rw") as db:
        db.put(b"k", b"v")

    mock_open.assert_called_with("/fake/path", read_only=False, create_if_missing=True)
