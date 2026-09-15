"""
Phase 5: full-unit ablation sweep and conditional-ablation test.

Sweep all 16, at the final checkpoint, to know if the interference 
pattern found early in training is still there once the model is fully 
converged, or was a transient "middle-step" of ahalf-trained network. 

If a unit is found whose ablation hurts its own class but helps
another, zero that unit only on inputs that are not
its own class, and check whether the helped class's accuracy actually
improves versus its baseline, with the unit's own class accuracy
unaffected by construction. 

Transition from binary match test to a continuous one.

Usage:
    python sweep.py runs/<run_name> [--step STEP]
If --step is omitted, uses the final checkpoint in the run.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import CLASS_NAMES, generate_dataset
from model import ColorNet
from interpret import (
    load_manifest, load_config, load_checkpoint_model,
    unit_class_selectivity, baseline_eval, ablate_unit_and_eval,
)


def rankdata_avg(a):
    """
    Rank an array with ties resolved by averaging, so a Pearson correlation 
    of these ranks gives the standard tie-corrected Spearman correlation.
    """
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
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[sorter[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x, y):
    """
    Spearman rank correlation between two equal-length vectors. 
    
    Returns None if either vector is constant (zero variance).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    rx = rankdata_avg(x)
    ry = rankdata_avg(y)
    return float(np.corrcoef(rx, ry)[0, 1])

def full_sweep(run_dir: Path, step: int, hidden_size: int, x_test, y_test):
    model = load_checkpoint_model(run_dir, step, hidden_size)
    baseline_acc, baseline_per_class = baseline_eval(model, x_test, y_test)

    probe_data = np.load(run_dir / "probe_set.npz", allow_pickle=True)
    probe_labels = probe_data["labels"]
    activations = np.load(run_dir / "checkpoints" / f"step_{step:07d}" / "activations.npy")
    class_means, preferred_class, selectivity = unit_class_selectivity(activations, probe_labels)

    """
    Raw activation only tells us how excited a unit is, which ignores whether that excitement
    actually pushes the network's decision anywhere. 

    Multiplying activation byoutgoing weight gives the unit's actual average
    contribution to each class's score.
    """

    w2 = model.fc2.weight.detach().numpy()
    rows = []
    for u in range(hidden_size):
        ab_acc, ab_per_class = ablate_unit_and_eval(model, u, x_test, y_test)
        per_class_drop = {
            c: (baseline_per_class[c] - ab_per_class[c])
            if baseline_per_class[c] is not None and ab_per_class[c] is not None else None
            for c in CLASS_NAMES
        }
        valid = {k: v for k, v in per_class_drop.items() if v is not None}
        worst_hit = max(valid, key=valid.get) if valid else None       # biggest positive drop (hurts most)
        best_helped = min(valid, key=valid.get) if valid else None     # most negative drop (helps most)
        pref = CLASS_NAMES[preferred_class[u]]

        # Correlate this unit's raw activation profile against its causal-impact profile
        activation_profile = [float(class_means[c, u]) for c in range(len(CLASS_NAMES))]
        causal_profile = [per_class_drop[c] if per_class_drop[c] is not None else 0.0 for c in CLASS_NAMES]
        corr = spearman_corr(activation_profile, causal_profile)

        # Direct logit attribution (activation x outgoing-weight)
        attribution_profile = [float(class_means[c, u] * w2[c, u]) for c in range(len(CLASS_NAMES))]
        attribution_corr = spearman_corr(attribution_profile, causal_profile)

        rows.append(dict(
            unit=u,
            preferred_class=pref,
            selectivity=float(selectivity[u]),
            overall_acc_drop=baseline_acc - ab_acc,
            worst_hit_class=worst_hit,
            worst_hit_drop=valid.get(worst_hit) if worst_hit else None,
            best_helped_class=best_helped,
            best_helped_gain=-valid.get(best_helped) if best_helped else None,  # positive = improvement
            matches_preferred_class=(worst_hit == pref),
            activation_causal_spearman=corr,
            attribution_causal_spearman=attribution_corr,
            activation_profile=activation_profile,
            attribution_profile=attribution_profile,
            causal_profile=causal_profile,
            per_class_drop=per_class_drop,
        ))

    return model, baseline_acc, baseline_per_class, rows


def conditional_ablation(model, unit_idx: int, own_class: str, X, y):
    """
    Conditional-ablation test: zero 'nit_idx' only on inputs whose
    true label is not 'own_class' then leave inputs of 'own_class' untouched.
    """
    own_idx = CLASS_NAMES.index(own_class)
    with torch.no_grad():
        logits, h = model(X, return_hidden=True)
        h_cond = h.clone()
        mask = (y != own_idx)  # True where allowed to ablate
        h_cond[mask, unit_idx] = 0.0
        logits_cond = model.fc2(h_cond)
        preds_cond = logits_cond.argmax(dim=1)

    overall_acc = (preds_cond == y).float().mean().item()
    per_class_acc = {}

    for c, cname in enumerate(CLASS_NAMES):
        cmask = y == c
        if cmask.sum() > 0:
            per_class_acc[cname] = (preds_cond[cmask] == y[cmask]).float().mean().item()
        else:
            per_class_acc[cname] = None
    return overall_acc, per_class_acc


def run(run_dir: str, step: int = None, n_test_eval: int = 10000, interference_threshold: float = 0.02):
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    config = load_config(run_dir)
    hidden_size = config["hidden_size"]

    if step is None:
        step = max(r["step"] for r in manifest)
        print(f"No --step given; using final checkpoint, step {step}")

    x_test_np, y_test_np = generate_dataset(n_test_eval, seed=config["seed"] + 2000000)
    x_test = torch.from_numpy(x_test_np)
    y_test = torch.from_numpy(y_test_np).long()

    model, baseline_acc, baseline_per_class, rows = full_sweep(run_dir, step, hidden_size, x_test, y_test)

    print(f"\nFull 16 unit ablation sweep at step {step}. Baseline test accuracy: {baseline_acc:.4f}\n")
    n_match = sum(1 for r in rows if r["matches_preferred_class"] and r["worst_hit_class"] is not None)
    print(f"[old binary metric] Units whose worst-hit class matches their activation preferred class: {n_match}/{hidden_size}")

    active_rows = [r for r in rows if not (r["selectivity"] == 0.0 and r["overall_acc_drop"] == 0.0)]
    act_corrs = [r["activation_causal_spearman"] for r in active_rows if r["activation_causal_spearman"] is not None]
    attr_corrs = [r["attribution_causal_spearman"] for r in active_rows if r["attribution_causal_spearman"] is not None]

    # Mean activation
    if act_corrs:
        mean_act_corr = sum(act_corrs) / len(act_corrs)
        print(f"[activation metric]  Mean activation vs causal Spearman over {len(act_corrs)} active units: "
              f"{mean_act_corr:.3f}")

    # Mean atribution
    if attr_corrs:
        mean_attr_corr = sum(attr_corrs) / len(attr_corrs)
        print(f"[attribution metric] Mean attribution vs causal Spearman over {len(attr_corrs)} active units: "
              f"{mean_attr_corr:.3f}\n")

    for r in rows:
        flag = "MATCH" if r["matches_preferred_class"] else "mismatch"
        act_str = f"{r['activation_causal_spearman']:+.2f}" if r["activation_causal_spearman"] is not None else " n/a"
        attr_str = f"{r['attribution_causal_spearman']:+.2f}" if r["attribution_causal_spearman"] is not None else " n/a"
        helped = f" helps {r['best_helped_class']} (+{r['best_helped_gain']:.3f})" if r["best_helped_gain"] and r["best_helped_gain"] > interference_threshold else ""
        print(f"  unit {r['unit']:2d}: pref={r['preferred_class']:6s} sel={r['selectivity']:.2f} "
              f"worst_hit={r['worst_hit_class']:6s} drop={r['worst_hit_drop']:.3f} [{flag}] "
              f"act_spearman={act_str} attr_spearman={attr_str}{helped}")

    """
    Find the strongest interference case: a unit that both is a real detector for that class and
    helps some other class by more than 'interference_threshold' when ablated.
    """

    interference_candidates = [
        r for r in rows
        if r["matches_preferred_class"] and r["best_helped_gain"] and r["best_helped_gain"] > interference_threshold
    ]
    interference_candidates.sort(key=lambda r: -r["best_helped_gain"])

    conditional_result = None
    if interference_candidates:
        top = interference_candidates[0]
        print(f"\nStrongest interference case: unit {top['unit']} (detector for {top['preferred_class']}), "
              f"ablating it helps {top['best_helped_class']} by {top['best_helped_gain']:.3f}.")
        print(f"Running conditional ablation test (zero unit {top['unit']} only on non"
              f"{top['preferred_class']} inputs)")

        cond_acc, cond_per_class = conditional_ablation(
            model, top["unit"], top["preferred_class"], x_test, y_test
        )
        helped_class = top["best_helped_class"]
        print(f"Overall accuracy: baseline={baseline_acc:.4f} conditional={cond_acc:.4f} "
              f"(delta {cond_acc - baseline_acc:+.4f})")
        print(f" {top['preferred_class']} accuracy: baseline={baseline_per_class[top['preferred_class']]:.4f} "
              f"conditional={cond_per_class[top['preferred_class']]:.4f} (should be unchanged by construction)")
        print(f" {helped_class} accuracy: baseline={baseline_per_class[helped_class]:.4f} "
              f"conditional={cond_per_class[helped_class]:.4f} "
              f"(delta {cond_per_class[helped_class] - baseline_per_class[helped_class]:+.4f})")

        conditional_result = dict(
            unit=top["unit"], own_class=top["preferred_class"], helped_class=helped_class,
            baseline_overall_acc=baseline_acc, conditional_overall_acc=cond_acc,
            baseline_per_class_acc=baseline_per_class, conditional_per_class_acc=cond_per_class,
        )
    else:
        print(f"\nNo unit both matched its preferred class and helped another class by more than "
              f"{interference_threshold:.2f} when ablated. No strong interferance case.")

    result = dict(
        step=step, baseline_acc=baseline_acc, baseline_per_class_acc=baseline_per_class,
        sweep=rows, n_match=n_match, hidden_size=hidden_size,
        interference_candidates=[r["unit"] for r in interference_candidates],
        conditional_ablation_result=conditional_result,
    )
    out_path = run_dir / f"phase5_sweep_step{step}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Path to runs/<run_name>")
    parser.add_argument("--step", type=int, default=None, help="Defaults to the final checkpoint.")
    parser.add_argument("--n-test-eval", type=int, default=10000)
    parser.add_argument("--interference-threshold", type=float, default=0.02,
                         help="Minimum accuracy gain on another class (when ablated) to count as interference.")
    args = parser.parse_args()
    run(args.run_dir, step=args.step, n_test_eval=args.n_test_eval,
        interference_threshold=args.interference_threshold)

# Program by Pedro Oubiña S. 2026