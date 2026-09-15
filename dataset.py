"""
Phase 0: Dataset definition.

Each RGB triple is labeled by nearest-neighbor distance (in RGB space) 
to a fixed and determined set of reference colors. 

"""

import numpy as np

# 11 basic color terms (Berlin & Kay's classic set), each with one representative of a RGB point.
REFERENCE_COLORS = {
    "red":    (220, 20, 20),
    "orange": (240, 130, 20),
    "yellow": (230, 220, 30),
    "green":  (30, 160, 60),
    "blue":   (30, 80, 200),
    "purple": (130, 40, 160),
    "pink":   (240, 150, 190),
    "brown":  (110, 70, 40),
    "black":  (15, 15, 15),
    "white":  (245, 245, 245),
    "gray":   (128, 128, 128),
}

CLASS_NAMES = list(REFERENCE_COLORS.keys())
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
REF_MATRIX = np.array([REFERENCE_COLORS[n] for n in CLASS_NAMES], dtype=np.float32)  # (11, 3)


def label_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Labels RGB points by nearest reference color (Euclidean distance in RGB space).
    rgb: (N, 3) array, values in [0, 255]
    returns: (N,) array of int class indices
    """
    diffs = rgb[:, None, :] - REF_MATRIX[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))
    return dists.argmin(axis=1)


def generate_dataset(n_samples: int, seed: int = 0):
    """
    Generate n_samples random RGB points, each with its label.
    Returns normalized inputs (0-1 range) and integer labels.
    """
    rng = np.random.default_rng(seed)
    rgb = rng.uniform(0, 255, size=(n_samples, 3)).astype(np.float32)
    labels = label_rgb(rgb)
    rgb_norm = rgb / 255.0
    return rgb_norm, labels


def class_balance_report(labels: np.ndarray):
    """
    Counts how many samples landed in each of the 11 classes to ensure a somewhat uniform distribuition
    and to provide data on the color distribuition.
    """
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    return {CLASS_NAMES[i]: int(counts[i]) for i in range(len(CLASS_NAMES))}

# 20000 samples generated, distribution is then printed
if __name__ == "__main__":
    X, y = generate_dataset(20000, seed=0)
    print("Dataset shape:", X.shape, y.shape)
    print("Class balance:", class_balance_report(y))

# Program by Pedro Oubiña S. 2026