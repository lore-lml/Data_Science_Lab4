import numpy as np
from random import randint
from math import sqrt


def _euclidean_distance(X, Y):
    return sqrt((X[1] - X[0])**2 + (Y[1] - Y[0])**2)


class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        self._max_x = None
        self._max_y = None

    def fit_predict(self, X):
        """Run the K-means clustering on x.
        :param X: input data points, array, shape = (N,C).
        :return: labels : array, shape = N.
        """
        self._build_centroids(X)
        for i in range(0, self.max_iter):
            self._assign_clusters(X)
            if self._recompute_centroids(X):
                break

        return self.labels, self.centroids

    def dump_to_file(self, filename):
        """Dump the evaluated labels to a CSV file."""
        with open(filename, "w", encoding='utf8') as fp:
            fp.write("ID, ClusterID\n")
            for i in range(self.n_clusters):
                for id in self.labels[i]:
                    fp.write(f"{id}, {i}\n")

    def _build_centroids(self, X):
        self._max_x = np.max(X[:, 0])
        self._max_y = np.max(X[:, 1])

        centroids = []
        for i in range(self.n_clusters):
            """l = [randint(0, self._max_x), randint(0, self._max_y)]
            centroids.append(l)"""
            l = [np.random.uniform(0, self._max_x), np.random.uniform(0, self._max_y)]
            centroids.append(l)


        self.centroids = np.array(centroids)
        # self._scatter_plot(X)

    def _assign_clusters(self, X):
        self.labels = [[] for i in range(self.n_clusters)]

        for i, (x, y) in enumerate(zip(X[:, 0], X[:, 1])):
            self._set_closer_cluster(x, y, i)

    def _set_closer_cluster(self, x, y, index_coord):
        distances = np.array([_euclidean_distance([x, x_c], [y, y_c]) for x_c, y_c in zip(self.centroids[:, 0],
                              self.centroids[:, 1])])

        index_centroid = np.argmin(distances)
        self.labels[index_centroid].append(index_coord)

    def _recompute_centroids(self, X):
        centroids = []

        for i in range(self.n_clusters):
            centroid_points = np.array(self.labels[i])
            if centroid_points.size > 0:
                x_c = np.mean(X[centroid_points, 0])
                y_c = np.mean(X[centroid_points, 1])
                centroids.append([x_c, y_c])
            else:
                centroids.append([self.centroids[i, 0], self.centroids[i, 1]])

        centroids = np.array(centroids)
        are_equal = np.array_equal(self.centroids, centroids)
        if not are_equal:
            self.centroids = centroids
        return are_equal
