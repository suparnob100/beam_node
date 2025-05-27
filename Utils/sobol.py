import numpy as np
from scipy.stats.qmc import Sobol
from scipy.spatial import cKDTree

def generate_sobol(dimensions, num_points, bounds):
    """
    Generates a Sobol sequence.
    """
    sobol = Sobol(d=dimensions)
    samples = sobol.random_base2(m=int(np.log2(num_points)))
    scaled_samples = np.empty_like(samples)
    
    for i in range(dimensions):
        lower, upper = bounds[i]
        scaled_samples[:, i] = samples[:, i] * (upper - lower) + lower
        
    return scaled_samples


def generate_sobol_with_exclusion(dimensions: int,
                                  num_points: int,
                                  bounds: list[tuple[float, float]],
                                  existing: np.ndarray | None = None,
                                  min_dist: float = 1e-6,
                                  oversample_factor: int = 4,
                                  scramble: bool = True) -> np.ndarray:
    """
    Generate exclusive sobol
    """
    tree_exist = cKDTree(existing) if (existing is not None and existing.size) else None
    accepted = []

    sob = Sobol(d=dimensions, scramble=scramble)
    m = int(np.ceil(np.log2(num_points * oversample_factor)))  # start size

    while len(accepted) < num_points:
        cand = sob.random_base2(m=m)

        for j, (lo, hi) in enumerate(bounds):
            cand[:, j] = cand[:, j] * (hi - lo) + lo

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