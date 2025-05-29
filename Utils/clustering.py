from sobol import generate_sobol_with_exclusion
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial import Delaunay
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

class Cluster:
    def __init__(self, data, params, normalize=True):
        self.data = data
        self.params = params
        self.normalize = normalize
        self.method = 'hierarchical'
        self.cluster_labels = None
        self.linkage_matrix = None
        self.flattened_data = np.array([m.flatten() for m in data])
        if self.normalize:
            self.flattened_data = StandardScaler().fit_transform(self.flattened_data)

    def find_clusters(self, method='hierarchical', **kwargs):
        self.method = method.lower()

        if self.method == 'hierarchical':
            self.linkage_matrix = linkage(self.flattened_data, method='ward')
            n_clusters = kwargs.get('n_clusters', 2)
            self.cluster_labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')

        elif self.method == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 2)
            self.cluster_labels = KMeans(n_clusters=n_clusters).fit_predict(self.flattened_data)

        elif self.method == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            self.cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(self.flattened_data)
            self.cluster_labels += 1

        elif self.method == 'agglomerative':
            n_clusters = kwargs.get('n_clusters', 2)
            self.cluster_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(self.flattened_data)
            self.cluster_labels += 1

        elif self.method == 'spectral':
            n_clusters = kwargs.get('n_clusters', 2)
            self.cluster_labels = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors').fit_predict(self.flattened_data)
            self.cluster_labels += 1

        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        self.n_clusters = kwargs.get('n_clusters', 2)

    def plot_dendrogram(self):
        if self.method != 'hierarchical' or self.linkage_matrix is None:
            raise ValueError("Dendrogram is only available for hierarchical clustering")
        plt.figure(figsize=(10, 7))
        plt.title('Hierarchical Clustering Dendrogram')
        dendrogram(self.linkage_matrix, leaf_rotation=90., leaf_font_size=8.)
        plt.xlabel('Matrix Index')
        plt.ylabel('Distance')
        plt.grid(True)
        plt.show()


    def plot_clusters_2d(self):
        if self.cluster_labels is None:
            raise ValueError("Run find_clusters() before plotting clusters.")

        params = np.array(self.params)

        if params.ndim == 1:
            x = params
            y = np.zeros_like(params)
            xlabel, ylabel = "Parameter", ""
        elif params.shape[1] == 1:
            x = params[:, 0]
            y = np.zeros_like(x)
            xlabel, ylabel = "Parameter", ""
        elif params.shape[1] >= 2:
            x = params[:, 0]
            y = params[:, 1]
            xlabel, ylabel = "Param 1", "Param 2"
        else:
            raise ValueError("Unsupported parameter shape for plotting.")

        unique_labels = np.unique(self.cluster_labels)
        n_labels = len(unique_labels)
        cmap = cm.get_cmap('tab10', n_labels)

        # Map cluster labels to index colors
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        color_indices = np.array([label_to_index[label] for label in self.cluster_labels])

        scatter = plt.scatter(x, y, c=color_indices, cmap=cmap, s=50, alpha=0.8, edgecolors='k')
        cbar = plt.colorbar(scatter, ticks=range(n_labels))
        cbar.ax.set_yticklabels([str(label) for label in unique_labels])
        cbar.set_label("Cluster Label")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"Clustered Parameter Space ({self.method.capitalize()})")
        plt.grid(True)
        plt.show()


    def compute_silhouette_score(self):
        if self.cluster_labels is None:
            raise ValueError("Run find_clusters() first.")
        
        labels = np.array(self.cluster_labels)
        unique = np.unique(labels)
        if len(unique) < 2:
            raise ValueError("Silhouette score needs at least 2 clusters.")

        return silhouette_score(self.flattened_data, labels)


    def find_best_k_by_silhouette(self, method='kmeans', k_range=range(2, 10), plot=True):
        scores = []

        for k in k_range:
            try:
                self.find_clusters(method=method, n_clusters=k)
                score = self.compute_silhouette_score()
                scores.append((k, score))
            except ValueError:
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

    def get_clustered_data(self):
        if self.cluster_labels is None:
            raise ValueError("Run find_clusters() before calling get_clustered_data().")

        n_clusters = int(np.max(self.cluster_labels))
        NLS_cl = [[] for _ in range(n_clusters)]
        param_cl = [[] for _ in range(n_clusters)]
        param_idx = [[] for _ in range(n_clusters)]

        for i, c in enumerate(self.cluster_labels):
            if c <= 0:
                continue  # Skip noise points (e.g., in DBSCAN)
            NLS_cl[c - 1].append(self.data[i])
            param_cl[c - 1].append(self.params[i])
            param_idx[c - 1].append(i)

        return NLS_cl, param_cl, param_idx

    def plot_delaunay(self, cluster_idx=0):

        params = np.array(self.get_clustered_data()[1][cluster_idx])
        if params.ndim == 1:
            params = params.reshape(-1, 1)
        
        n, d = params.shape

        if n < 3:
            print(f"[Delaunay] Cluster {cluster_idx} has too few points ({n}). Showing fallback plot.")
            plt.figure()
            if d == 1:
                plt.scatter(params[:, 0], np.zeros_like(params), s=50)
                plt.title(f"Cluster {cluster_idx} (1D - fallback)")
                plt.xlabel("Param")
                plt.yticks([])
            elif d == 2:
                plt.scatter(params[:, 0], params[:, 1], s=50)
                plt.title(f"Cluster {cluster_idx} (2D - fallback)")
                plt.xlabel("Param 1")
                plt.ylabel("Param 2")
            else:
                print("Unsupported dimension for plotting.")
                return None
            plt.grid(True)
            plt.show()
            return None

        if d == 1:
            plt.figure(figsize=(6, 2))
            plt.scatter(params[:, 0], np.zeros_like(params), s=50)
            plt.title(f"Cluster {cluster_idx} (1D)")
            plt.xlabel("Param")
            plt.yticks([])
            plt.grid(True)
            plt.show()
            return None

        elif d == 2:
            tri = Delaunay(params)
            plt.figure(figsize=(6, 5))
            plt.triplot(params[:, 0], params[:, 1], tri.simplices, color='gray')
            plt.plot(params[:, 0], params[:, 1], 'o', label=f"Cluster {cluster_idx}")
            plt.title(f"Delaunay Triangulation - Cluster {cluster_idx}")
            plt.xlabel("Param 1")
            plt.ylabel("Param 2")
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            plt.show()
            return tri

        else:
            print(f"[Delaunay] Cluster {cluster_idx} has {d}D parameters (unsupported).")
            return None

    def generate_from_cluster(self, cluster_idx, n_samples=10,
                          param_gen='gmm',
                          min_dist=1e-6, bandwidth=0.2,
                          oversample_factor=5):

        def is_in_hull(points, hull):
            return hull.find_simplex(points) >= 0
        
        if cluster_idx >= self.n_clusters:
            raise ValueError(f"cluster_idx must be smaller than n_clusters: {self.n_clusters}")

        param_cluster = np.array(self.get_clustered_data()[1][cluster_idx])

        n_cluster, d = param_cluster.shape

        if n_cluster < 3:
            print(f"[Cluster {cluster_idx}] Too few points ({n_cluster}) — fallback to Sobol.")
            bounds = [(param_cluster[:, i].min(), param_cluster[:, i].max()) for i in range(d)]
            new_params = generate_sobol_with_exclusion(
                dimensions=d,
                num_points=n_samples,
                bounds=bounds,
                existing=param_cluster,
                min_dist=min_dist,
                oversample_factor=oversample_factor,
                scramble=True
            )
        else:
            if param_gen == 'gmm':
                model = GaussianMixture(n_components=1).fit(param_cluster)
                sample_fn = lambda n: model.sample(n)[0]
            elif param_gen == 'kde':
                model = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(param_cluster)
                sample_fn = lambda n: model.sample(n)
            elif param_gen == 'sobol':
                bounds = [(param_cluster[:, i].min(), param_cluster[:, i].max()) for i in range(d)]
                new_params = generate_sobol_with_exclusion(
                    dimensions=d,
                    num_points=n_samples,
                    bounds=bounds,
                    existing=param_cluster,
                    min_dist=min_dist,
                    oversample_factor=oversample_factor,
                    scramble=True
                )
            else:
                raise ValueError("param_gen must be 'gmm', 'kde', or 'sobol'.")

            if param_gen in ['gmm', 'kde']:
                accepted = []
                if d <= 2:
                    hull = Delaunay(param_cluster)
                    while len(accepted) < n_samples:
                        samples = sample_fn(oversample_factor * (n_samples - len(accepted)))
                        mask = is_in_hull(samples, hull)
                        accepted.extend(samples[mask])
                else:
                    while len(accepted) < n_samples:
                        samples = sample_fn(n_samples - len(accepted))
                        accepted.extend(samples)
                new_params = np.array(accepted[:n_samples])


        return new_params

