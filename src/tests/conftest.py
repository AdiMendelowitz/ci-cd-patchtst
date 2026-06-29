"""Make sibling source packages importable during test collection.

test_experiments.py imports `generate_block_cov` (src/generators) and
`paired_stats` (src/analysis) as top-level modules. pytest only puts the test
directory on sys.path, so add the two sibling directories here.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _sub in ("generators", "analysis"):
    sys.path.insert(0, str(_SRC / _sub))
