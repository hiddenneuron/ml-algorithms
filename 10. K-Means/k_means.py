'''
K-Means

K-Means is an unsupervised learning algorithm used for clustering data into groups based on similarity. It groups data points into **k** clusters, where each point belongs to the cluster with the nearest mean.

Estimation: The data has **k** natural groupings, and each group can be represented by its center (mean).

Goal: To partition the data into **k** clusters such that the sum of squared distances between data points and their respective cluster centers is minimized.

Steps:

I. Initialization:

1. Choose the number of clusters (k).
2. Randomly initialize k cluster centers (called centroids). These can be chosen randomly from the dataset or initialized in other ways like k-means++.

II. Iterative Optimization (Repeat until convergence or max iterations):

1. Assign Step:

   * For each data point, compute the distance to each of the k centroids.
   * Assign each point to the nearest centroid (cluster).

2. Update Step:

   * Recalculate each centroid as the mean of all data points assigned to that cluster.

3. Repeat the assign and update steps until:

   * The assignments no longer change.
   * Or a maximum number of iterations is reached.
   * Or the centroid movements are very small (convergence).

III. Result:

* After convergence, the algorithm outputs:

  * The final cluster centers.
  * The cluster assignment for each data point.

Key Points:

* Simple and fast algorithm.
* Sensitive to initial placement of centroids (can lead to different results).
* Works well with spherical clusters of similar size.
* Doesn't work well with non-spherical or overlapping clusters or outliers.
* Choosing the right value of k is important (can use methods like the elbow method).

'''

import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # the centers (mean vector) for each cluster
        self.centroids = []


    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimize clusters
        for _ in range(self.max_iters):
            # assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels


    def _create_clusters(self, centroids):
        # assign the samples to the closest centroids
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx


    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between old and new centroids, for all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()


# Testing (copied directly from github)
if __name__ == "__main__":
    np.random.seed(42)
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, max_iters=150, plot_steps=True)
    y_pred = k.predict(X)

    k.plot()
