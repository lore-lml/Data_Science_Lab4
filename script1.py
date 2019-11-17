import numpy as np
import matplotlib.pyplot as plt
from Kmeans import KMeans

CLUSTERS = 15


def get_coordinates_from_file(path):
    return np.loadtxt(path, delimiter=',', skiprows=1, dtype='int64')


if __name__ == '__main__':
    X = get_coordinates_from_file("data_sets/2d_gauss_clusters.txt")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=90)
    ax.scatter(X[:, 0], X[:, 1])

    kmeans = KMeans(CLUSTERS)
    labels, centroids = kmeans.fit_predict(X)
    kmeans.dump_to_file("data_sets/kmean_result.txt")

    ax.scatter(centroids[:, 0], centroids[:, 1], marker="*")
    plt.show()