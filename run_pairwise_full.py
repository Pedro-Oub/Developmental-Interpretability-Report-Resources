"""
Runs pairwise_full.py across every run directory, pools all (pair,class)
rows into one combined dataset, and tests whether the per class
interaction term (attribution_u[c] * attribution_v[c]) actually predicts
the measured synergy when a pair is jointly ablated.

Usage:
    python run_pairwise_full.py

Optionally restrict to specific runs:
    python run_pairwise_full.py runs/run_seed1 runs/run_seed2
"""

import csv
import sys
from pathlib import Path

import numpy as np

from pairwise_full import full_pairwise_sweep, pearson_corr
from sweep import spearman_corr


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

    print(f"Running full pairwise sweep on {len(run_dirs)} run(s):\n")

    all_rows = []
    for rd in run_dirs:
        try:
            rows = full_pairwise_sweep(rd)
            all_rows.extend(rows)
        except Exception as e:
            print(f"FAILED on {rd}: {e}")

    if not all_rows:
        print("\nNo rows collected.")
        return

    combined_path = Path("runs") / "pairwise_full_combined.csv"
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    interaction = [r["interaction_term"] for r in all_rows]
    synergy = [r["synergy"] for r in all_rows]

    r_pearson = pearson_corr(interaction, synergy)
    r_spearman = spearman_corr(interaction, synergy)

    print(f"\n{'='*60}")
    print(f"Pooled dataset: {len(all_rows)} (pair, class) observations from {len(run_dirs)} runs")
    print(f"Saved: {combined_path}")
    print(f"\nInteraction term test synergy results:")
    print(f"Pearson r  = {r_pearson:.4f}" if r_pearson is not None else "  Pearson: undefined")
    print(f"Spearman r = {r_spearman:.4f}" if r_spearman is not None else "  Spearman: undefined")

    # Also compare against the single-unit prediction as a baseline
    additive = [r["attr_u"] + r["attr_v"] for r in all_rows]
    r_additive = pearson_corr(additive, synergy)
    print(f"\nSimple sum test synergy results:")
    print(f"  Pearson r  = {r_additive:.4f}" if r_additive is not None else "  Pearson: undefined")

if __name__ == "__main__":
    main()

# Program by Pedro Oubiña S. 2026