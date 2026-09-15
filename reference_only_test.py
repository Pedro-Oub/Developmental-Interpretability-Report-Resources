"""
Phase 5 methodology refinement: recompute activation/attribution profiles
using only the 11 unambiguous reference-color probe points, 
to test whether probe set ambiguity is limiting the 
activation vs causal and attribution vs causal correlations.

Usage:
    python reference_only_test.py
"""

import json
from pathlib import Path

import numpy as np

from dataset import CLASS_NAMES


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


# Print helper function for clarity
def summarize(label, values):
    values = [v for v in values if v is not None]
    n = len(values)
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5 if n > 1 else float("nan")
    se = std / (n ** 0.5) if n > 1 else float("nan")
    ci = 1.96 * se
    print(f"{label}: n={n} mean={mean:.4f} std={std:.4f} 95% CI [{mean-ci:.4f}, {mean+ci:.4f}]")
    return dict(label=label, n=n, mean=mean, std=std, ci95_low=mean - ci, ci95_high=mean + ci)


def probe_class_counts(run_dir: Path):
    d = np.load(run_dir / "probe_set.npz", allow_pickle=True)
    labels, kinds = d["labels"], d["kinds"]
    counts = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = labels == c
        counts[name] = dict(
            reference=int(((kinds == "reference") & mask).sum()),
            midpoint=int(((kinds == "midpoint") & mask).sum()),
            boundary=int(((kinds == "boundary") & mask).sum()),
            total=int(mask.sum()),
        )
    return counts


def main():
    run_dirs = sorted(p for p in Path("runs").iterdir() if p.is_dir() and (p / "config.json").exists())

    run_act_full, run_attr_full = [], []
    run_act_ref, run_attr_ref = [], []

    for rd in run_dirs:
        wpaths = sorted(rd.glob("fc2_weight_step*.json"))
        sweep_paths = sorted(rd.glob("phase5_sweep_step*.json"))
        if not wpaths or not sweep_paths:
            continue
        w2json = json.load(open(wpaths[-1]))
        w2 = np.array(w2json["w2"])
        step = w2json["step"]
        sweep = json.load(open(sweep_paths[-1]))

        probe = np.load(rd / "probe_set.npz", allow_pickle=True)
        labels, kinds = probe["labels"], probe["kinds"]
        ref_mask = kinds == "reference"
        ref_labels = labels[ref_mask]

        activations = np.load(rd / "checkpoints" / f"step_{step:07d}" / "activations.npy")
        ref_activations = activations[ref_mask]
        class_to_row = {int(c): i for i, c in enumerate(ref_labels)}

        acts_full, attrs_full, acts_ref, attrs_ref = [], [], [], []
        for r in sweep["sweep"]:
            if r["selectivity"] == 0.0 and r["overall_acc_drop"] == 0.0:
                continue
            if "attribution_profile" not in r:
                continue
            u = r["unit"]

            if r.get("activation_causal_spearman") is not None:
                acts_full.append(r["activation_causal_spearman"])
            if r.get("attribution_causal_spearman") is not None:
                attrs_full.append(r["attribution_causal_spearman"])

            act_refonly = [
                ref_activations[class_to_row[c], u] if c in class_to_row else 0.0
                for c in range(len(CLASS_NAMES))
            ]
            attr_refonly = [act_refonly[c] * w2[c, u] for c in range(len(CLASS_NAMES))]
            causal = [r["per_class_drop"].get(cn) or 0.0 for cn in CLASS_NAMES]

            ac = spearman_corr(act_refonly, causal)
            at = spearman_corr(attr_refonly, causal)
            if ac is not None:
                acts_ref.append(ac)
            if at is not None:
                attrs_ref.append(at)

        if acts_full: run_act_full.append(sum(acts_full) / len(acts_full))
        if attrs_full: run_attr_full.append(sum(attrs_full) / len(attrs_full))
        if acts_ref: run_act_ref.append(sum(acts_ref) / len(acts_ref))
        if attrs_ref: run_attr_ref.append(sum(attrs_ref) / len(attrs_ref))

    print("FULL PROBE SET (166 points)")
    s1 = summarize("Activation correlation (full probe set)", run_act_full)
    s2 = summarize("Attribution correlation (full probe set)", run_attr_full)
    print("\nREFERENCE COLORS ONLY (11 points)")
    s3 = summarize("Activation correlation (reference only)", run_act_ref)
    s4 = summarize("Attribution correlation (reference only)", run_attr_ref)

    counts = probe_class_counts(run_dirs[0])
    print("\nProbe points per class (full 166-point set, example run)")
    for name, c in counts.items():
        print(f"  {name:<8} reference={c['reference']} midpoint={c['midpoint']} "
              f"boundary={c['boundary']} total={c['total']}")

    results = dict(
        activation_full=s1, attribution_full=s2,
        activation_reference_only=s3, attribution_reference_only=s4,
        probe_class_counts=counts,
    )
    with open("runs/reference_only_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: runs/reference_only_test_results.json")

if __name__ == "__main__":
    main()

# Program by Pedro Oubiña S. 2026