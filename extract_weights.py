"""
Helper for the reference-color-only methodology test: pulls
just the final checkpoint's fc2 outgoing weight matrix out of each
run's weights.pt and saves it as json for ease of use.

Usage:
    python extract_weights.py

Optionally restrict to specific runs:
    python extract_weights.py runs/run_seed1 runs/run_seed2
"""

import json
import sys
from pathlib import Path

from interpret import load_config, load_checkpoint_model


# Find runs
def discover_run_dirs():
    runs_root = Path("runs")
    return sorted(
        p for p in runs_root.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    )

def extract_one_run(run_dir: Path):
    config = load_config(run_dir)
    hidden_size = config["hidden_size"]

    sweep_paths = sorted(run_dir.glob("phase5_sweep_step*.json"))

    if sweep_paths:
        step = int(sweep_paths[-1].stem.split("step")[-1])
    else:
        with open(run_dir / "manifest.csv") as f:
            import csv
            rows = list(csv.DictReader(f))
        step = max(int(r["step"]) for r in rows)

    model = load_checkpoint_model(run_dir, step, hidden_size)
    w2 = model.fc2.weight.detach().numpy().tolist()

    out_path = run_dir / f"fc2_weight_step{step}.json"
    with open(out_path, "w") as f:
        json.dump(dict(step=step, w2=w2), f)
    print(f"{run_dir.name}: saved {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dirs = [Path(p) for p in sys.argv[1:]]
    else:
        run_dirs = discover_run_dirs()

    for rd in run_dirs:
        try:
            extract_one_run(rd)
        except Exception as e:
            print(f"FAILED on {rd}: {e}")

# Program by Pedro Oubiña S. 2026