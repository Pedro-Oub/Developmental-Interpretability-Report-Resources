"""
Phase 5 methodology refinement: statistical significance filtering on
per class accuracy drops, before computing activation/attribution
correlations.

Every per class accuracy drop from single unit ablation has
been treated as a real causal effect, however small. 
A 1-2 point drop could easily be sampling noise rather than a genuine effect.
This filters out drops that aren't statistically distinguishable from zero 
(replacing them with 0) before correlating against activation/attribution
profiles, to test whether that changes the previously reported mean correlations.

The chosen test, unpaired z-test.
  SE = sqrt(p0(1-p0)/n + p1(1-p1)/n),  z = (p0 - p1) / SE
flagged significant if |z| > 1.96.

Usage:
    python significance_filter.py
"""

import glob # Ease of directory finding
import json
from pathlib import Path

import numpy as np

from dataset import generate_dataset, CLASS_NAMES


def rankdata_avg(a):
    a = np.asarray(a, dtype=float)
    n = len(a)
    sorter = np.argsort(a, kind="mergesort")
    ranks = np.empty(n)
    sorted_a = a[sorter]
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[sorter[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks

def spearman_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(np.corrcoef(rankdata_avg(x), rankdata_avg(y))[0, 1])


def per_class_counts(seed, n_test_eval=10000):
    _, y = generate_dataset(n_test_eval, seed=seed + 2000000)
    return {CLASS_NAMES[c]: int((y == c).sum()) for c in range(len(CLASS_NAMES))}


def filter_run(run_dir: Path, z_threshold=1.96): # Significance threshold
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    seed = config["seed"]
    counts = per_class_counts(seed)

    sweep_paths = sorted(run_dir.glob("phase5_sweep_step*.json"))
    with open(sweep_paths[-1]) as f:
        sweep = json.load(f)
    baseline_pc = sweep["baseline_per_class_acc"]

    rows = []
    for r in sweep["sweep"]:
        if r["selectivity"] == 0.0 and r["overall_acc_drop"] == 0.0:
            continue  # Skip dead units
        if "attribution_profile" not in r:
            continue

        act_profile = r["activation_profile"]
        attr_profile = r["attribution_profile"]

        causal_filtered = []
        n_significant = 0
        for c in CLASS_NAMES:
            p0 = baseline_pc[c]
            drop = r["per_class_drop"].get(c)
            drop = drop if drop is not None else 0.0
            p1 = min(max(p0 - drop, 0.0), 1.0)
            n = counts[c]
            if n == 0:
                causal_filtered.append(0.0)
                continue
            se = np.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n)
            z = drop / se if se > 0 else 0.0
            if abs(z) > z_threshold:
                causal_filtered.append(drop)
                n_significant += 1
            else:
                causal_filtered.append(0.0)

        act_corr = spearman_corr(act_profile, causal_filtered)
        attr_corr = spearman_corr(attr_profile, causal_filtered)
        rows.append(dict(
            run=run_dir.name, unit=r["unit"], n_significant_classes=n_significant,
            act_corr_filtered=act_corr, attr_corr_filtered=attr_corr,
            act_corr_unfiltered=r.get("activation_causal_spearman"),
            attr_corr_unfiltered=r.get("attribution_causal_spearman"),
        ))
    return rows

# Print helper function for clarity
def summarize(label, values):
    values = [v for v in values if v is not None]
    n = len(values)
    if n == 0:
        print(f"{label}: no data")
        return
    mean = sum(values) / n
    if n > 1:
        std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
        se = std / (n ** 0.5)
        ci = 1.96 * se
    else:
        std = se = ci = float("nan")
    print(f"{label}: n={n} mean={mean:.4f} std={std:.4f} 95% CI [{mean-ci:.4f}, {mean+ci:.4f}]")


def main():
    run_dirs = sorted(
        p for p in Path("runs").iterdir()
        if p.is_dir() and (p / "config.json").exists() and list(p.glob("phase5_sweep_step*.json"))
    )
    print(f"Filtering {len(run_dirs)} runs...\n")

    all_rows = []
    for rd in run_dirs:
        all_rows.extend(filter_run(rd))

    n_sig = [r["n_significant_classes"] for r in all_rows]
    print(f"Per-unit significant classes (out of 11): mean={sum(n_sig)/len(n_sig):.2f}, "
          f"min={min(n_sig)}, max={max(n_sig)}\n")

    by_run = {}
    for r in all_rows:
        by_run.setdefault(r["run"], []).append(r)

    # Filtering
    run_act_filtered, run_attr_filtered = [], []
    run_act_unfiltered, run_attr_unfiltered = [], []
    for run, rows in by_run.items():
        af = [r["act_corr_filtered"] for r in rows if r["act_corr_filtered"] is not None]
        tf = [r["attr_corr_filtered"] for r in rows if r["attr_corr_filtered"] is not None]
        au = [r["act_corr_unfiltered"] for r in rows if r["act_corr_unfiltered"] is not None]
        tu = [r["attr_corr_unfiltered"] for r in rows if r["attr_corr_unfiltered"] is not None]
        if af: run_act_filtered.append(sum(af) / len(af))
        if tf: run_attr_filtered.append(sum(tf) / len(tf))
        if au: run_act_unfiltered.append(sum(au) / len(au))
        if tu: run_attr_unfiltered.append(sum(tu) / len(tu))

    print("Before filtering (for reference)")
    summarize("Activation correlation (unfiltered)", run_act_unfiltered)
    summarize("Attribution correlation (unfiltered)", run_attr_unfiltered)
    print("\nAfter significance filtering")
    summarize("Activation correlation (filtered)", run_act_filtered)
    summarize("Attribution correlation (filtered)", run_attr_filtered)


if __name__ == "__main__":
    main()

# Program by Pedro Oubiña S. 2026