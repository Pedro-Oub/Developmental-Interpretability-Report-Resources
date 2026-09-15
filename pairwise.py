"""
Phase 5 extra: pairwise ablation.

The previous best single-unit predictor we have (direct logit
attribution) only explains about 72% of the variance in causal impact
(mean Spearman 0.850, rho^2 of 0.72).

One theory for the remaining 28%: some units might not be
independently responsible for a class at all: they might share the
job with a partner unit, such that ablating either ONE of them barely
matters (the partner covers up), but ablating both together
causes a large, disproportionate drop. Single unit ablation is
structurally blind to this.

Method, per run:
  1. Load the already computed full sweep (phase5_sweep_step*.json)
     reusing saved per unit attribution profiles and individual ablation results.
  2. Find the pair of active units whose attribution profiles are most
     correlated with each other (the most plausible "friendship" candidates).
  3. Ablate that pair jointly and compare the joint accuracy drop against 
     the sum of the two units' already-known individual drops.
  4. Synergy = joint_drop - (indiv_drop_u + indiv_drop_v).
     Synergy > 0 (joint hurts more than the sum of parts) is proof
     of redundant/backup coding. Synergy near 0 means the two
     units act independently (additive). Synergy < 0 would mean some
     kind of cancellation between them.

Usage:
    python pairwise.py runs/<run_name>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import CLASS_NAMES, generate_dataset
from interpret import load_config, load_checkpoint_model
from sweep import spearman_corr


def ablate_units_and_eval(model, unit_indices, x: torch.Tensor, y: torch.Tensor):
    with torch.no_grad():
        logits, h = model(x, return_hidden=True)
        h2 = h.clone()
        for idx in unit_indices:
            h2[:, idx] = 0.0
        logits2 = model.fc2(h2)
        preds = logits2.argmax(dim=1)

    overall_acc = (preds == y).float().mean().item()
    per_class_acc = {}
    for c, cname in enumerate(CLASS_NAMES):
        mask = y == c
        if mask.sum() > 0:
            per_class_acc[cname] = (preds[mask] == y[mask]).float().mean().item()
        else:
            per_class_acc[cname] = None
    return overall_acc, per_class_acc


def find_top_pair(sweep_json):
    """
    Among active units, find the pair whose attribution profiles are most positively
    correlated:the best candidate for a "friendship" pair.
    """
    rows = sweep_json["sweep"]
    active = [r for r in rows if not (r["selectivity"] == 0.0 and r["overall_acc_drop"] == 0.0)]

    best = None
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            u_row, v_row = active[i], active[j]
            if "attribution_profile" not in u_row or "attribution_profile" not in v_row:
                continue
            corr = spearman_corr(u_row["attribution_profile"], v_row["attribution_profile"])
            if corr is None:
                continue
            if best is None or corr > best[0]:
                best = (corr, u_row["unit"], v_row["unit"])
    return best

def run(run_dir: str, n_test_eval: int = 10000):
    run_dir = Path(run_dir)
    config = load_config(run_dir)
    hidden_size = config["hidden_size"]

    sweep_paths = sorted(run_dir.glob("phase5_sweep_step*.json"))
    if not sweep_paths:
        print(f"{run_dir}: no phase5_sweep_step*.json found, run sweep.py first.")
        return None
    with open(sweep_paths[-1]) as f:
        sweep_json = json.load(f)
    step = sweep_json["step"]

    pair = find_top_pair(sweep_json)
    if pair is None:
        print(f"{run_dir.name}: fewer than 2 active units with attribution profiles. Skipping.")
        return None
    pair_corr, u, v = pair

    rows_by_unit = {r["unit"]: r for r in sweep_json["sweep"]}
    baseline_acc = sweep_json["baseline_acc"]
    baseline_per_class = sweep_json["baseline_per_class_acc"]
    indiv_drop_u = rows_by_unit[u]["overall_acc_drop"]
    indiv_drop_v = rows_by_unit[v]["overall_acc_drop"]

    x_np, y_np = generate_dataset(n_test_eval, seed=config["seed"] + 2000000)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np).long()
    model = load_checkpoint_model(run_dir, step, hidden_size)

    joint_acc, joint_per_class = ablate_units_and_eval(model, [u, v], x, y)
    joint_drop = baseline_acc - joint_acc
    synergy = joint_drop - (indiv_drop_u + indiv_drop_v)

    per_class_joint_drop = {
        c: (baseline_per_class[c] - joint_per_class[c])
        if baseline_per_class[c] is not None and joint_per_class[c] is not None else None
        for c in CLASS_NAMES
    }
    per_class_indiv_sum = {
        c: (rows_by_unit[u]["per_class_drop"].get(c) or 0.0) + (rows_by_unit[v]["per_class_drop"].get(c) or 0.0)
        for c in CLASS_NAMES
    }
    per_class_synergy = {
        c: (per_class_joint_drop[c] - per_class_indiv_sum[c]) if per_class_joint_drop[c] is not None else None
        for c in CLASS_NAMES
    }
    worst_synergy_class = max(
        (c for c in per_class_synergy if per_class_synergy[c] is not None),
        key=lambda c: per_class_synergy[c], default=None
    )

    result = dict(
        run=run_dir.name, step=step, unit_u=u, unit_v=v, pair_attribution_corr=pair_corr,
        baseline_acc=baseline_acc, indiv_drop_u=indiv_drop_u, indiv_drop_v=indiv_drop_v,
        indiv_drop_sum=indiv_drop_u + indiv_drop_v, joint_drop=joint_drop, synergy=synergy,
        worst_synergy_class=worst_synergy_class,
        worst_synergy_value=per_class_synergy.get(worst_synergy_class) if worst_synergy_class else None,
        per_class_synergy=per_class_synergy,
    )
    out_path = run_dir / f"phase5_pairwise_u{u}_v{v}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"{run_dir.name}: pair=(unit {u}, unit {v}) attribution_corr={pair_corr:.3f} | "
          f"indiv drops: {indiv_drop_u:.3f} + {indiv_drop_v:.3f} = {indiv_drop_u+indiv_drop_v:.3f} "
          f"| joint drop: {joint_drop:.3f} | synergy: {synergy:+.3f}"
          + (f" | worst synergy class: {worst_synergy_class} ({result['worst_synergy_value']:+.3f})" if worst_synergy_class else ""))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Path to runs/<run_name>")
    parser.add_argument("--n-test-eval", type=int, default=10000)
    args = parser.parse_args()
    run(args.run_dir, n_test_eval=args.n_test_eval)

# Program by Pedro Oubiña S. 2026