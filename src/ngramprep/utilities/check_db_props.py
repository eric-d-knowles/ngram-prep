# ngramprep/utilities/check_db_props.py
from pathlib import Path
from ngramprep.common_db.api import open_db

# Your database path
SRC_DB = Path("/vast/edk202/NLP_corpora/Google_Books/20200217/eng/5gram_files/5grams.db")

def check_db_properties(db_path):
    """Check what RocksDB properties are available."""

    with open_db(db_path, mode="r") as db:
        properties_to_check = [
            "rocksdb.estimate-num-keys",
            "rocksdb.estimate-table-readers-mem",
            "rocksdb.size-all-mem-tables",
            "rocksdb.num-entries-active-mem-table",
            "rocksdb.estimate-live-data-size",
            "rocksdb.total-sst-files-size",
            "rocksdb.num-live-versions",
            "rocksdb.base-level",
            "rocksdb.estimate-pending-compaction-bytes"
        ]

        print("Available RocksDB Properties:")
        print("=" * 50)

        for prop in properties_to_check:
            try:
                value = db.get_property(prop)
                if value is not None:
                    if isinstance(value, int) and value > 1000:
                        print(f"✓ {prop}: {value:,}")
                    else:
                        print(f"✓ {prop}: {value}")
                else:
                    print(f"✗ {prop}: None")
            except Exception as e:
                print(f"✗ {prop}: Error - {e}")

# Run the check
check_db_properties(SRC_DB)
