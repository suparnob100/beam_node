from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import numpy as np

class Cluster:
    def __init__(self, data, params, seed, normalize=True):
        self.data = data
        self.params = np.array(params)
        if self.params.ndim == 1:
            self.params = self.params.reshape(-1, 1)  # force shape (N, 1) if flat
        self.normalize = normalize
        self.cluster_labels = None
        self.classifier = None
        self.n_clusters = None
        self.seed = seed

        self.flattened_data = np.array([m.flatten() for m in data])
        if self.normalize:
            self.flattened_data = StandardScaler().fit_transform(self.flattened_data)

    def find_clusters(self, n_clusters=3):
        self.n_clusters = n_clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed)
        self.cluster_labels = kmeans.fit_predict(self.flattened_data)
        self.classifier = KNeighborsClassifier(n_neighbors=1)
        self.classifier.fit(self.params, self.cluster_labels)

    def compute_silhouette_score(self):
        if self.cluster_labels is None:
            raise ValueError("Run find_clusters() first.")
        return silhouette_score(self.flattened_data, self.cluster_labels)
    
    def find_best_k_by_silhouette(self, k_range=range(2, 10), plot=True):
        scores = []

        for k in k_range:
            try:
                self.find_clusters(n_clusters=k)
                score = self.compute_silhouette_score()
                scores.append((k, score))
            except ValueError:
                scores.append((k, -1))

        if plot:
            ks, ss = zip(*scores)
            plt.figure(figsize=(6, 4))
            plt.plot(ks, ss, marker='o')
            plt.title('Silhouette Scores')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.grid(True)
            plt.show()

        for k, score in scores:
            print(f"k={k}, silhouette={score:.3f}")

        return scores

    def get_clustered_data(self):
        if self.cluster_labels is None:
            raise ValueError("Run find_clusters() first.")

        n_clusters = int(np.max(self.cluster_labels)) + 1
        NLS_cl = [[] for _ in range(n_clusters)]
        param_cl = [[] for _ in range(n_clusters)]
        param_idx = [[] for _ in range(n_clusters)]

        for i, c in enumerate(self.cluster_labels):
            NLS_cl[c].append(self.data[i])
            param_cl[c].append(self.params[i])
            param_idx[c].append(i)

        return NLS_cl, param_cl, param_idx

    def generate_points_in_cluster(self, cluster_idx, n_samples=10, min_dist=1e-3, oversample_factor=5):
        if self.classifier is None:
            raise ValueError("Run find_clusters() first.")

        param_dim = self.params.shape[1]
        bounds = [(self.params[:, i].min(), self.params[:, i].max()) for i in range(param_dim)]

        existing_points = np.array(self.get_clustered_data()[1][cluster_idx])
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

            # Check: classifier region
            predicted = self.classifier.predict(candidates)
            in_region = predicted == cluster_idx

            # Check: not too close to existing
            dists, _ = tree.query(candidates, k=1)
            far_enough = dists >= min_dist

            # Accept
            mask = in_region & far_enough
            accepted.extend(candidates[mask])

            attempts += 1

        new_params = np.array(accepted[:n_samples])
        return new_params


    def plot_decision_regions(self, resolution=300):
        if self.params.shape[1] == 1:
            x_min, x_max = self.params[:, 0].min() - 0.05, self.params[:, 0].max() + 0.05
            x_grid = np.linspace(x_min, x_max, resolution).reshape(-1, 1)
            y_pred = self.classifier.predict(x_grid)

            plt.figure(figsize=(8, 2))
            for cluster_label in np.unique(y_pred):
                mask = y_pred == cluster_label
                plt.fill_between(
                    x_grid[:, 0],
                    -0.5,
                    0.5,
                    where=mask,
                    alpha=0.3,
                    label=f"Cluster {cluster_label}"
                )

            plt.scatter(self.params[:, 0], np.zeros_like(self.params[:, 0]),
                        c=self.cluster_labels, cmap='tab10', edgecolors='k')
            plt.title("1D Cluster Decision Regions")
            plt.xlabel("Param")
            plt.yticks([])
            plt.grid(True)
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.show()
        elif self.params.shape[1] == 2:
            x_min, x_max = self.params[:, 0].min() - 0.05, self.params[:, 0].max() + 0.05
            y_min, y_max = self.params[:, 1].min() - 0.05, self.params[:, 1].max() + 0.05

            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, resolution),
                np.linspace(y_min, y_max, resolution)
            )

            Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.figure(figsize=(6, 6))
            plt.contourf(xx, yy, Z, alpha=0.4, cmap='Pastel1')
            plt.scatter(self.params[:, 0], self.params[:, 1],
                        c=self.cluster_labels, cmap='tab10', edgecolors='k')
            plt.title("2D Cluster Regions via KNN")
            plt.xlabel("Param 1")
            plt.ylabel("Param 2")
            plt.grid(True)
            plt.axis('equal')
            plt.show()
        else:
            print("[plot_decision_regions] Unsupported parameter dimension for plotting.")
