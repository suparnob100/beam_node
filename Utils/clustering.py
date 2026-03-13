from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import numpy as np

def _to_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def _scale_to_unit(x, bounds):
    """Scale each dimension to [0,1] given bounds for distance checks."""
    x = np.asarray(x, dtype=float)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    denom = np.maximum(hi - lo, 1e-12)
    return (x - lo) / denom

def _normalize_cluster_targets(min_points, n_clusters):
    """Normalize scalar/list/dict cluster targets into a dense int array."""
    if np.isscalar(min_points):
        return np.full(n_clusters, int(min_points), dtype=int)

    if isinstance(min_points, dict):
        values = min_points
        while values and all(isinstance(v, dict) for v in values.values()):
            values = next(iter(values.values()))

        targets = np.zeros(n_clusters, dtype=int)
        for cluster_idx in range(n_clusters):
            if cluster_idx not in values:
                raise KeyError(f"Missing min_points entry for cluster {cluster_idx}.")
            value = values[cluster_idx]
            if not np.isscalar(value):
                raise TypeError(
                    "Each min_points entry must be a scalar. "
                    "If you built the dict in a notebook, rerunning a cell may have nested it."
                )
            targets[cluster_idx] = int(value)
        return targets

    targets = np.asarray(min_points)
    if targets.ndim != 1 or len(targets) != n_clusters:
        raise ValueError(f"Expected {n_clusters} min_points values, received shape {targets.shape}.")
    return targets.astype(int)

class Cluster:
    def __init__(self, params, features, bounds=None, seed=42, normalize=True):
        self.params = np.array(params)
        if self.params.ndim == 1:
            self.params = self.params.reshape(-1, 1)

        self.bounds = bounds
        self.seed = seed
        self.n_clusters = None
        self.method = None
        self.cluster_labels = None
        self.classifier = None

        self.features = np.array([f.flatten() for f in features])
        if normalize:
            self.features = StandardScaler().fit_transform(self.features)

    def find_clusters(self, n_clusters=3, method='kmeans'):
        self.n_clusters = n_clusters
        self.method = method.lower()

        if self.method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=self.seed)
            self.cluster_labels = model.fit_predict(self.features)
        elif self.method == 'ward':
            self.linkage_matrix = linkage(self.features, method='ward')
            self.cluster_labels = fcluster(self.linkage_matrix, t=n_clusters, criterion='maxclust') - 1
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        self.classifier = KNeighborsClassifier(n_neighbors=1)
        self.classifier.fit(self.params, self.cluster_labels)

    def compute_silhouette_score(self):
        if self.cluster_labels is None:
            raise ValueError("Run find_clusters() first.")
        return silhouette_score(self.features, self.cluster_labels)

    def find_best_k_by_silhouette(self, k_range=range(2, 10), method='kmeans', plot=True):
        scores = []
        for k in k_range:
            try:
                self.find_clusters(n_clusters=k, method=method)
                score = self.compute_silhouette_score()
                scores.append((k, score))
            except Exception:
                scores.append((k, -1))

        if plot:
            ks, ss = zip(*scores)
            plt.figure(figsize=(6, 4))
            plt.plot(ks, ss, marker='o')
            plt.title(f'Silhouette Scores ({method})')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.grid(True)
            plt.show()

        for k, score in scores:
            print(f"k={k}, silhouette={score:.3f}")

        return scores

    def get_clustered_data(self, external_data, external_params):
        if self.cluster_labels is None:
            raise ValueError("Run find_clusters() first.")

        external_params = np.array(external_params)
        if external_params.ndim == 1:
            external_params = external_params.reshape(-1, 1)

        predicted = self.classifier.predict(external_params)
        n_clusters = int(np.max(self.cluster_labels)) + 1

        grouped_data = [[] for _ in range(n_clusters)]
        grouped_params = [[] for _ in range(n_clusters)]
        grouped_idx = [[] for _ in range(n_clusters)]

        for i, c in enumerate(predicted):
            grouped_data[c].append(external_data[i])
            grouped_params[c].append(external_params[i])
            grouped_idx[c].append(i)

        return grouped_data, grouped_params, grouped_idx



    def generate_points_in_cluster(
        self,
        cluster_idx,
        existing_points,
        n_samples=10,
        min_dist=1e-3,
        oversample_factor=5,
        max_attempts=1000,
        use_scaled_metric=True,
    ):
        """
        Improved sampler that enforces `min_dist` from BOTH the existing cluster points
        AND among newly accepted samples. Optionally performs distance checks in
        unit-scaled space (recommended).
        """
        # --- setup ---
        param_dim = self.params.shape[1]
        bounds = self.bounds if self.bounds is not None else [
            (self.params[:, i].min(), self.params[:, i].max()) for i in range(param_dim)
        ]

        existing_points = _to_2d(existing_points) if existing_points is not None else None
        accepted = []

        # Pre-scale existing points if using scaled metric
        if use_scaled_metric:
            existing_scaled = _scale_to_unit(existing_points, bounds) if existing_points is not None and len(existing_points) > 0 else None
            min_dist_eff = float(min_dist) / np.sqrt(param_dim)  # optional: make min_dist roughly per-dim
        else:
            existing_scaled = existing_points
            min_dist_eff = float(min_dist)

        attempts = 0
        while len(accepted) < n_samples and attempts < max_attempts:
            # Draw candidates uniformly over the bounds
            n_try = max(1, oversample_factor * (n_samples - len(accepted)))
            candidates = np.random.uniform(
                low=[b[0] for b in bounds],
                high=[b[1] for b in bounds],
                size=(n_try, param_dim),
            )

            # Keep only those classified as the desired cluster
            pred = self.classifier.predict(candidates)
            candidates = candidates[pred == cluster_idx]
            if candidates.size == 0:
                attempts += 1
                continue

            # Distance checks (scaled or raw)
            cand_scaled = _scale_to_unit(candidates, bounds) if use_scaled_metric else candidates

            # Build KDTree against (existing + accepted_so_far)
            if accepted:
                acc_arr = np.vstack(accepted)
                acc_scaled = _scale_to_unit(acc_arr, bounds) if use_scaled_metric else acc_arr
            else:
                acc_scaled = None

            # Merge existing + accepted for a single tree
            ref = None
            if existing_scaled is not None and len(existing_scaled) > 0:
                ref = existing_scaled
            if acc_scaled is not None and len(acc_scaled) > 0:
                ref = acc_scaled if ref is None else np.vstack([ref, acc_scaled])

            if ref is None:
                # No reference points yet, accept greedily while enforcing mutual spacing
                # by growing a temporary tree with each newly accepted point.
                tmp = []
                tmp_scaled = []
                for c, cs in zip(candidates, cand_scaled):
                    if not tmp_scaled:
                        tmp.append(c)
                        tmp_scaled.append(cs)
                    else:
                        tree = cKDTree(np.vstack(tmp_scaled))
                        d, _ = tree.query(cs, k=1)
                        if d >= min_dist_eff:
                            tmp.append(c)
                            tmp_scaled.append(cs)
                    if len(accepted) + len(tmp) >= n_samples:
                        break
                accepted.extend(tmp)
            else:
                # First check against reference
                tree = cKDTree(ref)
                d, _ = tree.query(cand_scaled, k=1)
                mask = d >= min_dist_eff
                filtered = candidates[mask]
                filtered_scaled = cand_scaled[mask]

                # Now enforce mutual spacing among the filtered ones themselves + accepted
                tmp = []
                tmp_scaled = []
                # Start a working tree with ref; we will incrementally add accepted points
                working = ref.copy()
                working_tree = cKDTree(working) if len(working) > 0 else None

                for c, cs in zip(filtered, filtered_scaled):
                    # Check distance to working set
                    ok = True
                    if working_tree is not None:
                        dist, _ = working_tree.query(cs, k=1)
                        ok = dist >= min_dist_eff
                    if ok:
                        tmp.append(c)
                        tmp_scaled.append(cs)
                        # update working set & tree
                        working = np.vstack([working, cs]) if working.size else cs[None, :]
                        working_tree = cKDTree(working)
                    if len(accepted) + len(tmp) >= n_samples:
                        break
                accepted.extend(tmp)

            attempts += 1

        if len(accepted) < n_samples:
            # Fallback: return whatever we have (or empty) rather than blocking
            pass

        return np.array(accepted[:n_samples])


    def balance_cluster_points(
        self,
        external_params,
        min_points,
        min_dist=1e-3,
        oversample_factor=5,
        use_scaled_metric=True,
        max_attempts=1000,
    ):
        """
        Ensure each cluster has at least `min_points` by sampling new points that:
        - are predicted to belong to the cluster, and
        - are at least `min_dist` away from existing points in that cluster AND from each other.

        Returns:
        dict: cluster_idx -> (new_points ndarray of shape [n_new, D])
        """
        external_params = _to_2d(external_params)
        cluster_map = self.classifier.predict(external_params)

        # Infer number of clusters either from fitted labels or predictions
        if hasattr(self, "cluster_labels") and self.cluster_labels is not None:
            n_clusters = int(np.max(self.cluster_labels)) + 1
        else:
            n_clusters = int(np.max(cluster_map)) + 1
        min_points = _normalize_cluster_targets(min_points, n_clusters)

        grouped = [[] for _ in range(n_clusters)]
        for i, label in enumerate(cluster_map):
            grouped[int(label)].append(external_params[i])

        new_param_dict = {
            cluster_idx: np.empty((0, external_params.shape[1]))
            for cluster_idx in range(n_clusters)
        }
        for cluster_idx, points in enumerate(grouped):
            points = np.vstack(points) if len(points) > 0 else np.empty((0, external_params.shape[1]))
            if len(points) >= min_points[cluster_idx]:
                continue
            n_needed = min_points[cluster_idx] - len(points)
            new_points = self.generate_points_in_cluster(
                cluster_idx=cluster_idx,
                existing_points=points,
                n_samples=n_needed,
                min_dist=min_dist,
                oversample_factor=oversample_factor,
                max_attempts=max_attempts,
                use_scaled_metric=use_scaled_metric,
            )
            new_param_dict[cluster_idx] = new_points

        return new_param_dict

    def plot_dendrogram(self):
        if self.method != 'ward':
            raise ValueError("Dendrogram is only available for 'ward' method.")
        if not hasattr(self, 'linkage_matrix'):
            raise RuntimeError("Run find_clusters() with method='ward' first.")

        plt.figure(figsize=(10, 6))
        dendrogram(self.linkage_matrix, leaf_rotation=90, leaf_font_size=8)
        plt.title("Hierarchical Clustering Dendrogram (Ward)")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_decision_regions(self, resolution=300):
        if self.params.shape[1] == 1:
            x_min, x_max = self.bounds[0] if self.bounds else (self.params[:, 0].min(), self.params[:, 0].max())
            x_grid = np.linspace(x_min, x_max, resolution).reshape(-1, 1)
            y_pred = self.classifier.predict(x_grid)
            plt.figure(figsize=(8, 2))
            for cluster_label in np.unique(y_pred):
                mask = y_pred == cluster_label
                plt.fill_between(x_grid[:, 0], -0.5, 0.5, where=mask, alpha=0.3, label=f"Cluster {cluster_label}")
            plt.scatter(self.params[:, 0], np.zeros_like(self.params[:, 0]),
                        c=self.cluster_labels, cmap='tab10', edgecolors='k')
            plt.title("1D Cluster Decision Regions")
            plt.xlabel("Param")
            plt.yticks([])
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        elif self.params.shape[1] == 2:
            if self.bounds:
                x_min, x_max = self.bounds[1]
                y_min, y_max = self.bounds[0]
            else:
                x_min, x_max = self.params[:, 0].min(), self.params[:, 0].max()
                y_min, y_max = self.params[:, 1].min(), self.params[:, 1].max()

            xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                                 np.linspace(y_min, y_max, resolution))
            Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            plt.figure(figsize=(6, 6))
            plt.contourf(xx, yy, Z, alpha=0.4, cmap='Pastel1')
            plt.scatter(self.params[:, 0], self.params[:, 1],
                        c=self.cluster_labels, cmap='tab10', edgecolors='k')
            plt.title("2D Cluster Regions")
            plt.xlabel("Param 1")
            plt.ylabel("Param 2")
            plt.grid(True)
            plt.axis('equal')
            plt.show()

        else:
            print("[plot_decision_regions] Unsupported parameter dimension.")

    def _infer_bounds_and_volume(self):
        """Return bounds array [(lo,hi),...], and total hypervolume."""
        if self.bounds is not None:
            bounds = list(self.bounds)
        else:
            bounds = [(self.params[:, i].min(), self.params[:, i].max())
                    for i in range(self.params.shape[1])]
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        vol = float(np.prod(np.maximum(hi - lo, 1e-12)))
        return bounds, vol

    def estimate_points_per_cluster_by_density(
        self,
        density: float | None = None,
        total_points: int | None = None,
        n_mc: int = 50000,
        use_scaled_space: bool = True,
        min_per_cluster: int = 0,
        max_per_cluster: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Monte-Carlo estimate of how many samples ("variables") to place in each cluster
        for a desired density or total number of samples.

        Args:
            density: points per unit hypervolume. If None, it will be inferred from `total_points`.
                    If `use_scaled_space=True`, density is w.r.t. the unit hypercube [0,1]^D.
            total_points: optional total number of points to allocate across clusters.
            n_mc: number of Monte-Carlo samples to estimate cluster hypervolumes.
            use_scaled_space: if True, estimate volumes in the unit cube (more stable),
                            else in raw parameter bounds.
            min_per_cluster: lower bound per cluster.
            max_per_cluster: optional upper bound per cluster.
            rng: optional NumPy Generator.

        Returns:
            counts: dict {cluster_idx: int}
            info:   dict with fractions, volumes, and raw float targets per cluster.
        """
        assert self.classifier is not None, "Call find_clusters() first to fit the classifier."

        rng = rng or np.random.default_rng(self.seed)
        D = self.params.shape[1]

        # Bounds and total volume
        bounds, vol_total = self._infer_bounds_and_volume()
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # Sample uniformly in either raw space or unit cube
        if use_scaled_space:
            U = rng.random((n_mc, D))
            X = lo + U * (hi - lo)           # map to raw for prediction
            vol_for_density = 1.0            # unit cube volume
        else:
            X = rng.uniform(lo, hi, size=(n_mc, D))
            vol_for_density = vol_total

        # Predict cluster for each MC sample
        labels = self.classifier.predict(X)
        K = int(np.max(labels)) + 1
        counts_mc = np.bincount(labels, minlength=K)
        frac = counts_mc / float(n_mc)

        # Cluster hypervolumes (in the space we picked for density)
        vols = frac * (1.0 if use_scaled_space else vol_total)

        # Decide density
        if density is None:
            if total_points is None:
                raise ValueError("Provide either `density` or `total_points`.")
            density = float(total_points) / vol_for_density

        # Real-valued targets, then integerize with Largest Remainder while satisfying bounds
        targets = density * vols  # float target per cluster (may not sum to total_points if density given)
        floors = np.floor(targets).astype(int)
        rema = targets - floors

        # If total_points given, adjust to hit it exactly
        if total_points is not None:
            deficit = int(total_points) - int(floors.sum())
            if deficit > 0:
                # give extra ones to largest remainders
                order = np.argsort(-rema)
                for i in order[:deficit]:
                    floors[i] += 1
            elif deficit < 0:
                # remove from smallest remainders (or where floors>0)
                order = np.argsort(rema)
                take = -deficit
                for i in order:
                    if take == 0:
                        break
                    if floors[i] > 0:
                        floors[i] -= 1
                        take -= 1

        # Enforce min/max with redistribution
        alloc = floors.copy()

        # First lift to minimums
        need = 0
        for k in range(K):
            if alloc[k] < min_per_cluster:
                need += (min_per_cluster - alloc[k])
                alloc[k] = min_per_cluster

        if max_per_cluster is not None:
            for k in range(K):
                if alloc[k] > max_per_cluster:
                    drop = alloc[k] - max_per_cluster
                    alloc[k] = max_per_cluster
                    need -= drop  # freeing capacity

        # Redistribute remaining need/capacity to respect totals if total_points specified
        if total_points is not None:
            diff = int(alloc.sum()) - int(total_points)
            if diff != 0:
                if diff > 0:
                    # too many → remove from clusters with smallest remainder first but above min
                    order = np.argsort(rema)
                    for i in order:
                        if diff == 0:
                            break
                        lim = min_per_cluster
                        while alloc[i] > lim and diff > 0:
                            alloc[i] -= 1
                            diff -= 1
                else:
                    # too few → add to clusters with largest remainder first but under max (if any)
                    order = np.argsort(-rema)
                    for i in order:
                        if diff == 0:
                            break
                        lim = max_per_cluster if max_per_cluster is not None else np.inf
                        while alloc[i] < lim and diff < 0:
                            alloc[i] += 1
                            diff += 1

        # Pack results
        out_counts = {k: int(alloc[k]) for k in range(K)}
        info = {
            "fraction": {k: float(frac[k]) for k in range(K)},
            "volume": {k: float(vols[k]) for k in range(K)},
            "targets_float": {k: float(targets[k]) for k in range(K)},
            "density_used": float(density),
            "use_scaled_space": bool(use_scaled_space),
            "estimated_total_volume": float(vol_for_density),
        }
        return out_counts, info
