"""
Example 03: Run Full Prefetching Benchmark
==========================================
Executes the full head-to-head comparison between all 4 baselines and saves
machine-readable JSON & CSV results.
"""

import sys
from pathlib import Path

# Add src to sys.path if running without pip install -e .
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from layeris.prefetch.benchmark import main as benchmark_main


if __name__ == "__main__":
    benchmark_main()
