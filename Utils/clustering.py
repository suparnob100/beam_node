from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import numpy as np


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

    def generate_points_in_cluster(self, cluster_idx, existing_points, n_samples=10, min_dist=1e-3, oversample_factor=5):
        param_dim = self.params.shape[1]
        bounds = self.bounds if self.bounds is not None else [
            (self.params[:, i].min(), self.params[:, i].max()) for i in range(param_dim)]

        tree = cKDTree(existing_points)
        accepted = []
        max_attempts = 1000
        attempts = 0

        while len(accepted) < n_samples and attempts < max_attempts:
            n_try = oversample_factor * (n_samples - len(accepted))
            candidates = np.random.uniform(
                low=[b[0] for b in bounds],
                high=[b[1] for b in bounds],
                size=(n_try, param_dim)
            )
            predicted = self.classifier.predict(candidates)
            in_region = predicted == cluster_idx
            dists, _ = tree.query(candidates, k=1)
            far_enough = dists >= min_dist
            mask = in_region & far_enough
            accepted.extend(candidates[mask])
            attempts += 1

        return np.array(accepted[:n_samples])

    def balance_cluster_points(self, external_params, min_points=16, min_dist=1e-3, oversample_factor=5):
        external_params = np.array(external_params)
        if external_params.ndim == 1:
            external_params = external_params.reshape(-1, 1)

        cluster_map = self.classifier.predict(external_params)
        n_clusters = int(np.max(self.cluster_labels)) + 1

        grouped = [[] for _ in range(n_clusters)]
        for i, label in enumerate(cluster_map):
            grouped[label].append(external_params[i])

        new_param_dict = {}
        for cluster_idx, points in enumerate(grouped):
            points = np.array(points)
            if len(points) >= min_points:
                continue
            n_needed = min_points - len(points)
            new_points = self.generate_points_in_cluster(
                cluster_idx=cluster_idx,
                existing_points=points,
                n_samples=n_needed,
                min_dist=min_dist,
                oversample_factor=oversample_factor
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
