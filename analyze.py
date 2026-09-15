"""
Phase 4: find interesting events in a completed run.

Loads every checkpoint's saved hidden-layer activations on the fixed
probe set and asks, at each step: how well separated are the 11 color
classes in hidden-activation space right now? We use the silhouette
score as that separation measure.

A "phase transition" candidate is a step where this separation score
jumps sharply: representations that were bundled suddenly reveal
themselves into distinguishable per class regions. 
Then cross-reference candidate steps against:
  - accuracy jumps (does the behavioral signal move at the same time?)
  - weight_delta_norm (was the change concentrated in that one step, or
    gradual and spread across many steps)

Output: a PNG with four stacked graphs (test accuracy, test loss,
silhouette separation, weight-delta norm all vs training step) and a
printed ranked list of transition candidates.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg") # Headless
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# Mainifest from train.py loading
def load_manifest(run_dir: Path):
    with open(run_dir / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            r[k] = float(r[k]) if k != "step" else int(r[k])
    return rows


# Load probe set
def load_probe_set(run_dir: Path):
    d = np.load(run_dir / "probe_set.npz", allow_pickle=True)
    return d["rgb_norm"], d["labels"], d["kinds"]


# Scikit silhouette score measure
def separation_score(activations: np.ndarray, labels: np.ndarray) -> float:
    try:
        return float(silhouette_score(activations, labels))
    except ValueError:
        return 0.0

# Separation score calculation and analysis
def analyze(run_dir: str, top_k: int = 5):
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    _, probe_labels, probe_kinds = load_probe_set(run_dir)

    steps, sep_scores = [], []
    for row in manifest:
        step = row["step"]
        act_path = run_dir / "checkpoints" / f"step_{step:07d}" / "activations.npy"
        activations = np.load(act_path)
        sep_scores.append(separation_score(activations, probe_labels))
        steps.append(step)

    steps = np.array(steps)
    sep_scores = np.array(sep_scores)
    test_acc = np.array([row["test_acc"] for row in manifest])
    test_loss = np.array([row["test_loss"] for row in manifest])
    weight_delta = np.array([row["weight_delta_norm"] for row in manifest])

    # Biggest single-checkpoint jumps in separation score
    sep_jumps = np.diff(sep_scores)
    order = np.argsort(-sep_jumps)[:top_k]
    candidates = []
    for i in order:
        step = int(steps[i + 1])
        candidates.append(dict(
            step=step,
            sep_jump=float(sep_jumps[i]),
            sep_before=float(sep_scores[i]),
            sep_after=float(sep_scores[i + 1]),
            acc_before=float(test_acc[i]),
            acc_after=float(test_acc[i + 1]),
            weight_delta_norm=float(weight_delta[i + 1]),
        ))

    # Save results for Phase 5
    with open(run_dir / "transition_candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    axes[0].plot(steps, test_acc, color="#2563eb")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title(f"Run: {run_dir.name}")

    axes[1].plot(steps, test_loss, color="#dc2626")
    axes[1].set_ylabel("test loss")

    axes[2].plot(steps, sep_scores, color="#16a34a")
    axes[2].set_ylabel("silhouette\n(class separation)")

    axes[3].plot(steps, weight_delta, color="#7c3aed")
    axes[3].set_ylabel("weight delta\nnorm")
    axes[3].set_xlabel("training step")

    for c in candidates:
        for ax in axes:
            ax.axvline(c["step"], color="gray", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = run_dir / "phase4_overview.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved plot: {out_path}")
    print(f"Saved candidates: {run_dir / 'transition_candidates.json'}")
    print("\nTop transition candidates (by silhouette score jump):")
    for c in candidates:
        print(
            f"step {c['step']:5d}: separation {c['sep_before']:.3f} {c['sep_after']:.3f} "
            f"(+{c['sep_jump']:.3f}), acc {c['acc_before']:.3f}  {c['acc_after']:.3f}, "
            f"weight_delta_norm={c['weight_delta_norm']:.4f}"
        )

    return candidates


# Runs full run analysis
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Path to runs/<run_name>")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    analyze(args.run_dir, top_k=args.top_k)

# Program by Pedro Oubiña S. 2026