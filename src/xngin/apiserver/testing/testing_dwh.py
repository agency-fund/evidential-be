"""Utilities for the management of the local testing database.

This corresponds to the "testing" config specified in xngin.testing.settings.json.
"""

import hashlib
from pathlib import Path

TESTING_DWH_RAW_DATA = Path(__file__).parent.parent / "testdata/testing_dwh.csv.zst"


def compact_hash(path: Path):
    """Computes a hash of the input CSV so that we can determine whether to re-create the test warehouse."""
    h = hashlib.blake2b(digest_size=2)
    with open(path, "rb") as source:
        h.update(source.read())
    with open(__file__, "rb") as source:
        h.update(source.read())
    return int.from_bytes(h.digest())
