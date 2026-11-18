"""Run the full Project Atlas pipeline on synthetic data.

Usage (from the project folder, after `pip install -e .`):
    python scripts/run_demo.py            # full demo, about one to two minutes on a laptop
    python scripts/run_demo.py --fast     # smaller simulations

Results are written to data/outputs/: forecasts.csv (Spec §7.9), risk.csv (Spec §8.8), portfolio.csv,
signal_tests.csv and research_registry.csv (Spec §9.1), monitoring.csv (Spec §9.6), stress tests and a
short demo_summary.md report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works without installation too

from atlas.pipeline.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))
