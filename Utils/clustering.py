from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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

        # Ensure 2D for plotting
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
        """
        Computes the silhouette score of the current clustering.
        """
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
