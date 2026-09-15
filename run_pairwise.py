"""
Runs pairwise.py across every run directory and aggregates the synergy 
results across seeds, the same way aggregate.py does for the single unit sweep.

Usage:
    python run_pairwise.py

Optionally restrict to specific runs:
    python run_pairwise.py runs/run_seed1 runs/run_seed2
"""

import sys
from pathlib import Path

from pairwise import run as run_pairwise


# Find runs
def discover_run_dirs():
    runs_root = Path("runs")
    return sorted(
        p for p in runs_root.iterdir()
        if p.is_dir() and (p / "manifest.csv").exists()
    )


def main():
    if len(sys.argv) > 1:
        run_dirs = [Path(p) for p in sys.argv[1:]]
    else:
        run_dirs = discover_run_dirs()

    print(f"Running pairwise ablation on {len(run_dirs)} run(s):\n")

    results = []
    for rd in run_dirs:
        try:
            r = run_pairwise(str(rd))
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"FAILED on {rd}: {e}")

    if not results:
        print("\nNo results to aggregate")
        return

    synergies = [r["synergy"] for r in results]
    n = len(synergies)
    mean_syn = sum(synergies) / n
    n_positive = sum(1 for s in synergies if s > 0.01)   # Clearly superadditive
    n_near_zero = sum(1 for s in synergies if -0.01 <= s <= 0.01)
    n_negative = sum(1 for s in synergies if s < -0.01)

    if n > 1:
        var = sum((s - mean_syn) ** 2 for s in synergies) / (n - 1)
        std = var ** 0.5
        se = std / (n ** 0.5)
        ci95 = 1.96 * se
    else:
        std = se = ci95 = float("nan") # Nothing

    print(f"\n{'='*60}")
    print(f"Aggregate over {n} runs:")
    print(f"Mean synergy: {mean_syn:+.4f} (std={std:.4f}, SE={se:.4f}, "
          f"95% CI [{mean_syn-ci95:+.4f}, {mean_syn+ci95:+.4f}])")
    print(f"Superadditive: {n_positive}/{n} runs")
    print(f"Independent: {n_near_zero}/{n} runs")
    print(f"Subadditive/cancelling: {n_negative}/{n} runs")
    print(f"Per run synergies: {[round(s,3) for s in synergies]}")


if __name__ == "__main__":
    main()

# Program by Pedro Oubiña S. 2026