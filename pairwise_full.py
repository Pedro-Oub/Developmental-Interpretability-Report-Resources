"""
Phase 5 extra: full pairwise sweep + interaction term test.

pairwise.py tested exactly one pair per run (the pair whose attribution
profiles looked most alike) and found real synergy, but an after
check found that "how alike two units' profiles look" only weakly
predicts how big their synergy actually is (Pearson r = 0.167 across the
25 top-pairs).

This tests a different, more targeted candidate feature: instead of one
number summarizing a whole pair's similarity, use a per class interaction
term: unit u's attribution to class c times unit v's attribution to
class c, then ask whether that predicts the per class synergy when the
pair is jointly ablated.

To get enough data to test this as proper as possible, this sweeps 
all pairs of active units in a run, producing one row per
(pair, class) instead of one row per run. Pooled across 25 runs, this
gives thousands of data points instead of only 25.

Usage:
    python pairwise_full.py runs/<run_name>
"""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from dataset import CLASS_NAMES, generate_dataset
from interpret import load_config, load_checkpoint_model


def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def full_pairwise_sweep(run_dir, n_test_eval: int = 10000):
    run_dir = Path(run_dir)
    config = load_config(run_dir)
    hidden_size = config["hidden_size"]

    sweep_paths = sorted(run_dir.glob("phase5_sweep_step*.json"))
    if not sweep_paths:
        print(f"{run_dir}: no phase5_sweep_step*.json, run sweep.py first")
        return []
    with open(sweep_paths[-1]) as f:
        sweep_json = json.load(f)
    step = sweep_json["step"]
    baseline_per_class = sweep_json["baseline_per_class_acc"]

    rows_by_unit = {r["unit"]: r for r in sweep_json["sweep"]}
    active_units = [
        r["unit"] for r in sweep_json["sweep"]
        if (not(r["selectivity"] == 0.0 and r["overall_acc_drop"] == 0.0)
        and "attribution_profile" in r)
    ]

    x_np, y_np = generate_dataset(n_test_eval, seed=config["seed"] + 2000000)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np).long()
    y_np_arr = y.numpy()

    model = load_checkpoint_model(run_dir, step, hidden_size)
    with torch.no_grad():
        _, h = model(x, return_hidden=True)  # Compute hidden activations once, reuse for every pair

    out_rows = []
    pairs = list(itertools.combinations(active_units, 2))
    for u, v in pairs:
        with torch.no_grad():
            h2 = h.clone()
            h2[:, u] = 0.0
            h2[:, v] = 0.0
            preds = model.fc2(h2).argmax(dim=1).numpy() # Predictions

        for c, cname in enumerate(CLASS_NAMES):
            mask = y_np_arr == c
            if mask.sum() == 0:
                continue
            joint_acc_c = float((preds[mask] == c).mean())
            joint_drop_c = baseline_per_class[cname] - joint_acc_c

            indiv_u = rows_by_unit[u]["per_class_drop"].get(cname) or 0.0
            indiv_v = rows_by_unit[v]["per_class_drop"].get(cname) or 0.0
            indiv_sum = indiv_u + indiv_v
            synergy = joint_drop_c - indiv_sum

            attr_u = rows_by_unit[u]["attribution_profile"][c]
            attr_v = rows_by_unit[v]["attribution_profile"][c]
            interaction_term = attr_u * attr_v

            out_rows.append(dict(
                run=run_dir.name, unit_u=u, unit_v=v, class_name=cname,
                attr_u=attr_u, attr_v=attr_v, interaction_term=interaction_term,
                indiv_drop_u=indiv_u, indiv_drop_v=indiv_v, indiv_sum=indiv_sum,
                joint_drop=joint_drop_c, synergy=synergy,
            ))

    out_path = run_dir / "phase5_pairwise_full.csv"
    if out_rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)

    print(f"{run_dir.name}: {len(active_units)} active units,  {len(pairs)} pairs"
          f"{len(out_rows)} (pair,class) rows saved to {out_path}")
    return out_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Path to runs/<run_name>")
    parser.add_argument("--n-test-eval", type=int, default=10000)
    args = parser.parse_args()
    full_pairwise_sweep(args.run_dir, n_test_eval=args.n_test_eval)

# Program by Pedro Oubiña S. 2026