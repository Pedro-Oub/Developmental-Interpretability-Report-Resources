"""
Phase 5: interpret a flagged transition moment.

Takes a candidate step from Phase 4 (as default the one with the single
biggest test-accuracy jump, for manifest.csv) and asks three questions:

  1. Which hidden units changed the most, going into that step?
     (weight-level: change in each unit's incoming RGB weights and
     outgoing per-class weights between the checkpoint before and after)
  2. Do those units correspond to something nameable? Is a unit
     consistently high for "red" probe points and low for everything
     else, and so a "redness detector", or is it superposition?
  3. Does ablating a flagged unit specifically hurt the classes 
     it seems to encode, more than other classes?

Usage:
    python interpret.py runs/<run_name> [--step STEP] [--top-units N]
If --step is omitted, uses the step with the largest single-checkpoint
test-accuracy jump (independent of, and a useful cross-check against,
Phase 4's silhouette-based candidates).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dataset import CLASS_NAMES
from model import ColorNet
from dataset import generate_dataset


def load_manifest(run_dir: Path):
    with open(run_dir / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            r[k] = float(r[k]) if k != "step" else int(r[k])
    return rows


def load_config(run_dir: Path):
    with open(run_dir / "config.json") as f:
        return json.load(f)


# Reconstruct ColorNet and load ckpt weights for comparison
def load_checkpoint_model(run_dir: Path, step: int, hidden_size: int):
    model = ColorNet(hidden_size=hidden_size)
    state = torch.load(run_dir / "checkpoints" / f"step_{step:07d}" / "weights.pt",
                        map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def pick_biggest_accuracy_jump(manifest):
    steps = [r["step"] for r in manifest]
    accs = [r["test_acc"] for r in manifest]
    jumps = [(steps[i + 1], accs[i + 1] - accs[i]) for i in range(len(steps) - 1)]
    jumps.sort(key=lambda x: -x[1])
    return jumps[0][0]


# Find the checkpoint step at/just-before or at/just-after target_step
def nearest_checkpoint_step(manifest, target_step, before: bool):
    steps = sorted(r["step"] for r in manifest)
    if before:
        candidates = [s for s in steps if s <= target_step]
        return max(candidates) if candidates else steps[0]
    else:
        candidates = [s for s in steps if s >= target_step]
        return min(candidates) if candidates else steps[-1]


def per_unit_weight_delta(model_before: ColorNet, model_after: ColorNet):
    """
    For each hidden unit, combine how much its incoming weights (from
    the 3 RGB inputs + bias) changed and how much its outgoing
    weights (to the 11 class logits) changed, into one per-unit score.
    """
    # Before
    w1_before = model_before.fc1.weight.detach().numpy()  # (hidden, 3)
    b1_before = model_before.fc1.bias.detach().numpy()    # (hidden,)
    w2_before = model_before.fc2.weight.detach().numpy()  # (11, hidden)

    # After
    w1_after = model_after.fc1.weight.detach().numpy()
    b1_after = model_after.fc1.bias.detach().numpy()
    w2_after = model_after.fc2.weight.detach().numpy()

    incoming_before = np.concatenate([w1_before, b1_before[:, None]], axis=1)  # (hidden, 4)
    incoming_after = np.concatenate([w1_after, b1_after[:, None]], axis=1)
    incoming_delta = np.linalg.norm(incoming_after - incoming_before, axis=1)  # (hidden,)

    outgoing_delta = np.linalg.norm(w2_after - w2_before, axis=0)  # (hidden,)

    total_delta = incoming_delta + outgoing_delta
    return incoming_delta, outgoing_delta, total_delta


def unit_class_selectivity(activations: np.ndarray, labels: np.ndarray):
    """
    For each hidden unit, compute its mean activation per class over the
    probe set. Return means, preferred_class and selectivity, which is the 
    measure of how clear of a color indicator a unit is.
    """
    n_classes = len(CLASS_NAMES)
    hidden_size = activations.shape[1]
    class_means = np.zeros((n_classes, hidden_size))
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_means[c] = activations[mask].mean(axis=0)

    preferred_class = class_means.argmax(axis=0)  
    sorted_means = np.sort(class_means, axis=0) # Ascending
    top = sorted_means[-1]
    second = sorted_means[-2]
    selectivity = (top - second) / (np.abs(top) + 1e-6)

    return class_means, preferred_class, selectivity


def ablate_unit_and_eval(model: ColorNet, unit_idx: int, X: torch.Tensor, y: torch.Tensor):
    """
    Zero a single hidden unit activation (post ReLU) for every input,
    then run the rest of the forward pass as normal. Returns overall
    accuracy and per-class accuracy with that unit ablated.
    """
    with torch.no_grad():
        logits, h = model(X, return_hidden=True)
        h_ablated = h.clone()
        h_ablated[:, unit_idx] = 0.0
        logits_ablated = model.fc2(h_ablated)
        preds = logits_ablated.argmax(dim=1)

    overall_acc = (preds == y).float().mean().item()
    per_class_acc = {}
    for c in range(len(CLASS_NAMES)):
        mask = y == c
        if mask.sum() > 0:
            per_class_acc[CLASS_NAMES[c]] = (preds[mask] == y[mask]).float().mean().item()
        else:
            per_class_acc[CLASS_NAMES[c]] = None
    return overall_acc, per_class_acc


def baseline_eval(model: ColorNet, X: torch.Tensor, y: torch.Tensor):
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
    overall_acc = (preds == y).float().mean().item()
    per_class_acc = {}
    for c in range(len(CLASS_NAMES)):
        mask = y == c
        if mask.sum() > 0:
            per_class_acc[CLASS_NAMES[c]] = (preds[mask] == y[mask]).float().mean().item()
        else:
            per_class_acc[CLASS_NAMES[c]] = None
    return overall_acc, per_class_acc

def interpret(run_dir: str, step: int = None, top_units: int = 3, n_test_eval: int = 5000):
    """
    Pick the target step, load the before/after checkpoint pair, rank
    units by weight change, take the top 'top_units' (default 3), look up each one's apparent
    preferred class from activations. Then ablate each of those top units on a fresh 5,000-
    point test set and record, per unit, which class was hurt the most, and whether that "worst hit"
    class matches the unit's activation-based "preferred class" guess.
    """
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    config = load_config(run_dir)
    hidden_size = config["hidden_size"]

    if step is None:
        step = pick_biggest_accuracy_jump(manifest)
        print(f"No --step given; using biggest test-accuracy jump, step {step}")

    step_before = nearest_checkpoint_step(manifest, step, before=True)
    step_after = nearest_checkpoint_step(manifest, step, before=False)
    if step_before == step_after:
        earlier = [r["step"] for r in manifest if r["step"] < step_after]
        step_before = max(earlier) if earlier else step_after
    print(f"Comparing checkpoint step {step_before} (before) -> step {step_after} (after)")

    model_before = load_checkpoint_model(run_dir, step_before, hidden_size)
    model_after = load_checkpoint_model(run_dir, step_after, hidden_size)

    # Which units changed the most
    incoming_delta, outgoing_delta, total_delta = per_unit_weight_delta(model_before, model_after)
    ranked_units = np.argsort(-total_delta)

    # Do the top-changed units correspond to something nameable
    probe_data = np.load(run_dir / "probe_set.npz", allow_pickle=True)
    probe_labels = probe_data["labels"]
    activations_after = np.load(run_dir / "checkpoints" / f"step_{step_after:07d}" / "activations.npy")
    class_means, preferred_class, selectivity = unit_class_selectivity(activations_after, probe_labels)

    unit_report = []
    for rank, u in enumerate(ranked_units[:top_units]):
        unit_report.append(dict(
            unit=int(u),
            rank=rank,
            weight_delta_total=float(total_delta[u]),
            weight_delta_incoming=float(incoming_delta[u]),
            weight_delta_outgoing=float(outgoing_delta[u]),
            preferred_class=CLASS_NAMES[preferred_class[u]],
            selectivity=float(selectivity[u]),
        ))

    print("\nTop changed units at this transition:")
    for r in unit_report:
        print(f"  unit {r['unit']:2d} (rank {r['rank']}): weight_delta={r['weight_delta_total']:.3f} "
              f"(in={r['weight_delta_incoming']:.3f}, out={r['weight_delta_outgoing']:.3f}) | "
              f"preferred class: {r['preferred_class']} | selectivity={r['selectivity']:.3f}")

    # Ablation
    x_test_np, y_test_np = generate_dataset(n_test_eval, seed=config["seed"] + 1_000_000)
    x_test = torch.from_numpy(x_test_np)
    y_test = torch.from_numpy(y_test_np).long()

    baseline_acc, baseline_per_class = baseline_eval(model_after, x_test, y_test)

    ablation_report = []
    for r in unit_report:
        u = r["unit"]
        ab_acc, ab_per_class = ablate_unit_and_eval(model_after, u, x_test, y_test)
        per_class_drop = {
            cname: (baseline_per_class[cname] - ab_per_class[cname])
            if baseline_per_class[cname] is not None and ab_per_class[cname] is not None else None
            for cname in CLASS_NAMES
        }
        # Class with the biggest accuracy drop from ablating this unit
        valid_drops = {k: v for k, v in per_class_drop.items() if v is not None}
        worst_hit_class = max(valid_drops, key=valid_drops.get) if valid_drops else None

        ablation_report.append(dict(
            unit=u,
            preferred_class=r["preferred_class"],
            overall_acc_after_ablation=ab_acc,
            overall_acc_drop=baseline_acc - ab_acc,
            per_class_drop=per_class_drop,
            worst_hit_class=worst_hit_class,
            worst_hit_drop=valid_drops.get(worst_hit_class) if worst_hit_class else None,
            matches_preferred_class=(worst_hit_class == r["preferred_class"]),
        ))

    print(f"\nBaseline test accuracy (step {step_after}, no ablation): {baseline_acc:.4f}")
    print("\nAblation results:")
    for a in ablation_report:
        match_str = "MATCHES preferred class" if a["matches_preferred_class"] else "does NOT match preferred class"
        print(f"  unit {a['unit']:2d} (preferred={a['preferred_class']}): "
              f"overall acc {baseline_acc:.4f} -> {a['overall_acc_after_ablation']:.4f} "
              f"(drop {a['overall_acc_drop']:.4f}); worst-hit class = {a['worst_hit_class']} "
              f"(drop {a['worst_hit_drop']:.4f}) [{match_str}]")

    result = dict(
        step_target=step, step_before=step_before, step_after=step_after,
        unit_report=unit_report, baseline_acc=baseline_acc,
        baseline_per_class_acc=baseline_per_class, ablation_report=ablation_report,
    )

    out_path = run_dir / f"phase5_interpretation_step{step_after}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Path to runs/<run_name>")
    parser.add_argument("--step", type=int, default=None,
                         help="Target step to interpret. Defaults to the biggest test-accuracy jump.")
    parser.add_argument("--top-units", type=int, default=3,
                         help="How many top-changed hidden units to analyze and ablate.")
    parser.add_argument("--n-test-eval", type=int, default=5000,
                         help="Size of freshly-generated test set used for ablation eval.")
    args = parser.parse_args()
    interpret(args.run_dir, step=args.step, top_units=args.top_units, n_test_eval=args.n_test_eval)

# Program by Pedro Oubiña S. 2026