"""
Experiment runner: for each given seed, runs train.py -> analyze.py ->
sweep.py in sequence, so a full replication is one command per seed. 
(Ease of use and step automatization)

Usage:
    python run_experiment.py --seeds 1 2 3 4
    python run_experiment.py --seeds 1 2 3 4 --total-steps 3000
"""

import argparse
import subprocess
import sys


# Single run
def run_one(seed: int, total_steps: int, checkpoint_every: int):
    run_name = f"run_seed{seed}"
    print(f"\n{'='*60}\nSeed {seed} runs/{run_name}\n{'='*60}")

    subprocess.run([
        sys.executable, "train.py",
        "--seed", str(seed),
        "--run-name", run_name,
        "--total-steps", str(total_steps),
        "--checkpoint-every", str(checkpoint_every),
    ], check=True)

    subprocess.run([sys.executable, "analyze.py", f"runs/{run_name}"], check=True)
    subprocess.run([sys.executable, "sweep.py", f"runs/{run_name}"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--total-steps", type=int, default=3000)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    # Combine all runs
    for s in args.seeds:
        run_one(s, args.total_steps, args.checkpoint_every)

    print("\nAll seeds complete.")

# Program by Pedro Oubiña S. 2026