"""Make sibling source packages importable during test collection.

test_experiments.py imports `models` (src), `generate_ar1_grid` and `generate_block_cov`
(src/generators), and `paired_stats` (src/analysis) as top-level modules.
pytest only puts the test directory on sys.path, so add src and its two
subdirectories here.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
for _sub in ("", "generators", "analysis"):
    sys.path.insert(0, str(_SRC / _sub))