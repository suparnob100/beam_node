"""Sobol quasi-random sequence generation with optional exclusion zones."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats.qmc import Sobol
from scipy.spatial import cKDTree


def generate_sobol(
    dimensions: int,
    num_points: int,
    bounds: list[tuple[float, float]],
) -> NDArray:
    """Generate a scaled Sobol sequence.

    Parameters
    ----------
    dimensions : int
        Number of parameter dimensions.
    num_points : int
        Number of points (rounded down to a power of 2).
    bounds : list of (lower, upper)
        Scaling bounds per dimension.

    Returns
    -------
    NDArray [num_points, dimensions]
    """
    sobol = Sobol(d=dimensions)
    samples = sobol.random_base2(m=int(np.log2(num_points)))

    # Vectorized scaling (no per-dimension loop)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return samples * (hi - lo) + lo


def generate_sobol_with_exclusion(
    dimensions: int,
    num_points: int,
    bounds: list[tuple[float, float]],
    existing: NDArray | None = None,
    min_dist: float = 1e-6,
    oversample_factor: int = 4,
    scramble: bool = True,
) -> NDArray:
    """Generate a Sobol sequence excluding points near *existing*.

    Parameters
    ----------
    dimensions : int
        Number of parameter dimensions.
    num_points : int
        Desired number of output points.
    bounds : list of (lower, upper)
        Scaling bounds per dimension.
    existing : NDArray or None
        Existing points to avoid (within *min_dist*).
    min_dist : float
        Minimum Euclidean distance from existing points.
    oversample_factor : int
        Oversampling multiplier per retry.
    scramble : bool
        Whether to apply Owen scrambling.

    Returns
    -------
    NDArray [num_points, dimensions]
    """
    tree_exist = cKDTree(existing) if (existing is not None and existing.size) else None
    accepted: list[list[float]] = []

    sob = Sobol(d=dimensions, scramble=scramble)
    m = int(np.ceil(np.log2(num_points * oversample_factor)))

    # Pre-compute scaling arrays
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    while len(accepted) < num_points:
        cand = sob.random_base2(m=m)
        # Vectorized scaling
        cand = cand * (hi - lo) + lo

        if tree_exist is not None:
            good = tree_exist.query(cand, k=1)[0] >= min_dist
            cand = cand[good]

        if accepted:
            tree_new = cKDTree(np.asarray(accepted))
            good = tree_new.query(cand, k=1)[0] >= min_dist
            cand = cand[good]

        accepted.extend(cand.tolist())
        m += 1

    return np.asarray(accepted[:num_points])
