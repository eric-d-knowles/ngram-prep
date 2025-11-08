"""Tests for partition caching functionality."""

import tempfile
import shutil
from pathlib import Path

from ngramkit.tracking import PartitionCache, PartitionCacheKey, WorkUnit


def test_partition_cache_basic():
    """Test basic partition caching operations."""
    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = PartitionCache(Path(tmpdir))

        # Create test cache key
        cache_key = PartitionCacheKey(
            db_path="/test/db.db",
            num_units=10,
            samples_per_worker=1000,
            num_sampling_workers=4
        )

        # Create test work units
        work_units = [
            WorkUnit(unit_id="unit_1", start_key=None, end_key=b'\x10'),
            WorkUnit(unit_id="unit_2", start_key=b'\x10', end_key=b'\x20'),
            WorkUnit(unit_id="unit_3", start_key=b'\x20', end_key=None),
        ]

        # Initially cache should not exist
        assert not cache.exists(cache_key), "Cache should not exist initially"

        # Save to cache
        cache.save(cache_key, work_units, metadata={"test": "value"})

        # Now cache should exist
        assert cache.exists(cache_key), "Cache should exist after save"

        # Load from cache
        loaded_units = cache.load(cache_key)
        assert loaded_units is not None, "Should load from cache"
        assert len(loaded_units) == len(work_units), "Should have same number of units"

        # Verify work unit details
        for orig, loaded in zip(work_units, loaded_units):
            assert orig.unit_id == loaded.unit_id
            assert orig.start_key == loaded.start_key
            assert orig.end_key == loaded.end_key

        # Get cache info
        info = cache.get_info(cache_key)
        assert info is not None, "Should have cache info"
        assert info["num_work_units"] == 3
        assert info["metadata"]["test"] == "value"

        # List all caches
        all_caches = cache.list_all()
        assert len(all_caches) == 1, "Should have one cache entry"

        # Clear specific cache
        cache.clear(cache_key)
        assert not cache.exists(cache_key), "Cache should not exist after clear"

        print("✓ All basic cache tests passed")


def test_partition_cache_multiple_entries():
    """Test multiple cache entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = PartitionCache(Path(tmpdir))

        # Create multiple cache keys and entries
        for i in range(3):
            cache_key = PartitionCacheKey(
                db_path=f"/test/db{i}.db",
                num_units=10 + i,
                samples_per_worker=1000,
                num_sampling_workers=4
            )

            work_units = [
                WorkUnit(unit_id=f"unit_{i}_{j}", start_key=None, end_key=None)
                for j in range(5)
            ]

            cache.save(cache_key, work_units)

        # Verify all exist
        all_caches = cache.list_all()
        assert len(all_caches) == 3, "Should have three cache entries"

        # Clear all
        cache.clear()
        all_caches = cache.list_all()
        assert len(all_caches) == 0, "Should have no cache entries after clear all"

        print("✓ Multiple entries test passed")


def test_partition_cache_key_hash_uniqueness():
    """Test that different cache keys produce different hashes."""
    key1 = PartitionCacheKey(
        db_path="/test/db.db",
        num_units=10,
        samples_per_worker=1000,
        num_sampling_workers=4
    )

    key2 = PartitionCacheKey(
        db_path="/test/db.db",
        num_units=20,  # Different
        samples_per_worker=1000,
        num_sampling_workers=4
    )

    key3 = PartitionCacheKey(
        db_path="/test/other.db",  # Different
        num_units=10,
        samples_per_worker=1000,
        num_sampling_workers=4
    )

    hash1 = key1.to_hash()
    hash2 = key2.to_hash()
    hash3 = key3.to_hash()

    assert hash1 != hash2, "Different num_units should produce different hash"
    assert hash1 != hash3, "Different db_path should produce different hash"
    assert hash2 != hash3, "All hashes should be unique"

    print("✓ Cache key hash uniqueness test passed")


if __name__ == "__main__":
    test_partition_cache_basic()
    test_partition_cache_multiple_entries()
    test_partition_cache_key_hash_uniqueness()
    print("\n✅ All partition cache tests passed!")
