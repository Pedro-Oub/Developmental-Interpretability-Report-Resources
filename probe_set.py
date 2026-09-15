"""
Phase 2, Part 1: fixed probe set for tracking hidden-layer activations
over the course of training.

Probe set is seeded and reused at every checkpoint, so
activation differences across checkpoints reflect changes in the model,
not changes in the input. 

It includes:

1. The 11 reference colors themselves, easy cases.

2. Pairwise midpoints between reference colors, ambiguous cases.

3. Points sampled uniformly from RGB space whose distance to the 
   nearest reference color is almost tied with the distance to the 
   second-nearest reference color, sitting near true decision boundaries.

Each probe point is tagged with its type ("reference", "midpoint", "boundary") 
and a label (true class under the same nearest reference rule used for training) for later analysis. 

"""

import itertools
import numpy as np

from dataset import REFERENCE_COLORS, CLASS_NAMES, REF_MATRIX, label_rgb

# Reference probe points
def _reference_probes():
    rgb = REF_MATRIX.copy()  # (11, 3), 0-255
    kinds = ["reference"] * len(rgb)
    return rgb, kinds

# Midpoint color probe points
def _midpoint_probes():
    pairs = list(itertools.combinations(range(len(CLASS_NAMES)), 2))
    mids = np.array(
        [(REF_MATRIX[i] + REF_MATRIX[j]) / 2.0 for i, j in pairs],
        dtype=np.float32,
    )
    kinds = ["midpoint"] * len(mids)
    return mids, kinds


def _boundary_probes(n_probes: int = 100, n_candidates: int = 200_000, seed: int = 42):
    """
    Sample many random RGB points and keep the n_probes with the smallest
    margin between nearest and second-nearest reference color distance.
    These lie close to the true decision boundaries of the labeling rule. 
    This is more informative than midpoints alone, since midpoints between 
    two references can still be closer to a third reference entirely.
    """
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(0, 255, size=(n_candidates, 3)).astype(np.float32)

    diffs = candidates[:, None, :] - REF_MATRIX[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))  # (N, 11)
    sorted_dists = np.sort(dists, axis=1)
    margin = sorted_dists[:, 1] - sorted_dists[:, 0]  # nearest vs. second-nearest

    idx = np.argsort(margin)[:n_probes] # Most ambiguous points
    boundary = candidates[idx]
    kinds = ["boundary"] * len(boundary)
    return boundary, kinds


def build_probe_set(n_boundary: int = 100, seed: int = 42):
    """
    returns:
      rgb_norm: (N, 3) float32, normalized to [0, 1] (matches training input)
      labels:   (N,)   int64, ground-truth class index under the same rule 
                       used to generate training data
      kinds:    (N,)   array of descriptive strings: "reference"  "midpoint"  "boundary"
    
    166 total points
    """
    ref_rgb, ref_kinds = _reference_probes()
    mid_rgb, mid_kinds = _midpoint_probes()
    bnd_rgb, bnd_kinds = _boundary_probes(n_probes=n_boundary, seed=seed)

    rgb = np.concatenate([ref_rgb, mid_rgb, bnd_rgb], axis=0)
    kinds = np.array(ref_kinds + mid_kinds + bnd_kinds)
    labels = label_rgb(rgb)
    rgb_norm = rgb / 255.0

    return rgb_norm.astype(np.float32), labels.astype(np.int64), kinds


if __name__ == "__main__":
    rgb_norm, labels, kinds = build_probe_set()
    print("Probe set shape:", rgb_norm.shape)
    unique, counts = np.unique(kinds, return_counts=True)
    print("Kind breakdown:", dict(zip(unique, counts)))
    print("Class coverage:", {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(len(CLASS_NAMES))})

# Program by Pedro Oubiña S. 2026