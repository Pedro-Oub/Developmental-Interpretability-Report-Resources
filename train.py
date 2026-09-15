"""
Phase 2 part 2 and phase 3: training loop.

Trains ColorNet while "tasting" it at a fixed step cadence.

At each checkpoint step we save three things:
  1. Full weight tensors (state_dict).
  2. Hidden-layer activations on the fixed probe set (probe_set.py)
     thhis is the primary "did something real happen" signal.
  3. Accuracy/loss on a held-out test set.

We also record the L2 norm of the weight change since the previous
checkpoint to use later in Phases 4 and 5: weight-delta magnitude helps 
show whether it was concentrated in a few weights or diffuse across many.

Output layout (per run):

    runs/<run_name>/
        config.json                  # hyperparameters, for reproducibility
        probe_set.npz                # rgb_norm, labels, kinds 
        manifest.csv                 # one row per checkpoint: step, losses, accs, weight_delta_norm
        checkpoints/
            step_0000000/
                weights.pt            # model.state_dict()
                activations.npy       # (n_probe, hidden_size) hidden-layer activations
                metrics.json          # step, train_loss, train_acc, test_loss, test_acc, weight_delta_norm
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Program imports
from dataset import generate_dataset
from model import ColorNet
from probe_set import build_probe_set

# Cleaner model eval that saves compute and memory
def evaluate(model, X, y, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = criterion(logits, y).item()
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    model.train()
    return loss, acc


# Flatten parameter tensors in the model
def weight_vector(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


# "Tasting" function, 
def save_checkpoint(run_dir, step, model, probe_X, prev_weight_vec, train_loss, train_acc,
                     test_loss, test_acc):
    # Checkpoint directory creation
    ckpt_dir = run_dir / "checkpoints" / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Saving weights
    torch.save(model.state_dict(), ckpt_dir / "weights.pt")

    # Hidden activations on fixed probe set
    model.eval()
    with torch.no_grad():
        _, hidden = model(probe_X, return_hidden=True)
    model.train()
    np.save(ckpt_dir / "activations.npy", hidden.numpy())

    # Weight delta norm vs previous checkpoint
    curr_weight_vec = weight_vector(model)
    if prev_weight_vec is None:
        weight_delta_norm = 0.0
    else:
        weight_delta_norm = (curr_weight_vec - prev_weight_vec).norm().item()

    metrics = {
        "step": step,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "weight_delta_norm": weight_delta_norm,
    }
    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, curr_weight_vec


# Model training
def train(
    run_name: str = None,
    total_steps: int = 3000,
    checkpoint_every: int = 25,
    batch_size: int = 64,
    lr: float = 0.05,
    hidden_size: int = 16,
    n_train: int = 20000,
    n_test: int = 5000,
    n_boundary_probes: int = 100,
    seed: int = 0,
    runs_dir: str = "runs",
):
    # Reproductibility seeding
    torch.manual_seed(seed)
    np.random.seed(seed)

    if run_name is None:
        run_name = time.strftime("run_%Y%m%d_%H%M%S")

    run_dir = Path(runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameter saving
    config = dict(
        run_name=run_name, total_steps=total_steps, checkpoint_every=checkpoint_every,
        batch_size=batch_size, lr=lr, hidden_size=hidden_size, n_train=n_train,
        n_test=n_test, n_boundary_probes=n_boundary_probes, seed=seed,
    )
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Separate seeds for train and test so they don't overlap
    x_train_np, y_train_np = generate_dataset(n_train, seed=seed)
    x_test_np, y_test_np = generate_dataset(n_test, seed=seed + 1_000_000)
    x_train = torch.from_numpy(x_train_np)
    y_train = torch.from_numpy(y_train_np).long()
    x_test = torch.from_numpy(x_test_np)
    y_test = torch.from_numpy(y_test_np).long()

    # Fixed probe set, saved once so phases 4 and 5 can reload it exactly
    probe_rgb, probe_labels, probe_kinds = build_probe_set(n_boundary=n_boundary_probes, seed=seed + 42)
    np.savez(run_dir / "probe_set.npz", rgb_norm=probe_rgb, labels=probe_labels, kinds=probe_kinds)
    probe_X = torch.from_numpy(probe_rgb)

    model = ColorNet(hidden_size=hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    manifest_rows = []
    prev_weight_vec = None

    n = x_train.shape[0]
    step = 0
    epoch_perm = torch.randperm(n)
    cursor = 0

    # Checkpoint at initialization (step 0) before any training happens as comparison baseline
    while step <= total_steps:
        if step % checkpoint_every == 0 or step == total_steps:
            train_loss, train_acc = evaluate(model, x_train, y_train, criterion)
            test_loss, test_acc = evaluate(model, x_test, y_test, criterion)
            metrics, prev_weight_vec = save_checkpoint(
                run_dir, step, model, probe_X, prev_weight_vec,
                train_loss, train_acc, test_loss, test_acc,
            )
            manifest_rows.append(metrics)

        if step == total_steps:
            break

        # Reshuffle when we run out of the epoch, so every step still sees an unbiased random batch
        if cursor + batch_size > n:
            epoch_perm = torch.randperm(n)
            cursor = 0
        idx = epoch_perm[cursor:cursor + batch_size]
        cursor += batch_size

        xb, yb = x_train[idx], y_train[idx]
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        step += 1

    # Manifest for quick later without opening every metrics.json
    with open(run_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    final = manifest_rows[-1]
    print(f"Run '{run_name}' complete. {len(manifest_rows)} checkpoints saved to {run_dir}")
    print(f"Final: train_acc={final['train_acc']:.4f} test_acc={final['test_acc']:.4f}")
    return run_dir


# CLI indicators
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=3000)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--n-boundary-probes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-dir", type=str, default="runs")
    args = parser.parse_args()

    train(
        run_name=args.run_name, total_steps=args.total_steps,
        checkpoint_every=args.checkpoint_every, batch_size=args.batch_size,
        lr=args.lr, hidden_size=args.hidden_size, n_train=args.n_train,
        n_test=args.n_test, n_boundary_probes=args.n_boundary_probes,
        seed=args.seed, runs_dir=args.runs_dir,
    )

# Program by Pedro Oubiña S. 2026