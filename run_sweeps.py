"""
Sweeps runner: calls sweep.py analysis on every existing run
directory.

No retraining happens here, it just re-runs the Phase 5 full unit
ablation sweep against checkpoints that already exist on disk, picking
up the updated Spearman correlation metric added in sweep.py.

Usage:
    python run_sweeps.py
    s
Optionally restrict to specific run directories:
    python run_sweeps.py runs/run_seed1 runs/run_seed2
"""

import sys
from pathlib import Path

from sweep import run as run_sweep


# Find runs
def discover_run_dirs():
    runs_root = Path("runs")
    return sorted(
        p for p in runs_root.iterdir()
        if p.is_dir() and (p / "manifest.csv").exists()
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dirs = [Path(p) for p in sys.argv[1:]]
    else:
        run_dirs = discover_run_dirs()

    print(f"Running sweep.py on {len(run_dirs)} run(s):")
    for rd in run_dirs:
        print(f"  {rd}")

    failures = []
    for rd in run_dirs:
        print(f"\n{'='*60}\n{rd}\n{'='*60}")
        try:
            run_sweep(str(rd))
        except Exception as e:
            print(f"FAILED on {rd}: {e}")
            failures.append(str(rd))

    print(f"\nDone {len(run_dirs) - len(failures)}/{len(run_dirs)} succeeded.")
    if failures:
        print(f"Failed: {failures}")

# Program by Pedro Oubiña S. 2026