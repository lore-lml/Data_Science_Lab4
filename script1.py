import numpy as np
import matplotlib.pyplot as plt
from Kmeans import KMeans

CLUSTERS = 15


def get_coordinates_from_file(path):
    return np.loadtxt(path, delimiter=',', skiprows=1)


if __name__ == '__main__':
    files_in = ["data_sets/2d_gauss_clusters.txt", "data_sets/chameleon_clusters.txt"]
    files_out = ["data_sets/kmean_result_2d_gauss.txt", "data_sets/kmean_result_chamaleon.txt"]

    for i in range(2):
        X = get_coordinates_from_file(files_in[i])

        kmeans = KMeans(CLUSTERS)
        labels, centroids = kmeans.fit_predict(X)
        kmeans.dump_to_file(files_out[i])

        _, ax = plt.subplots(figsize=(6, 6), dpi=90)
        ax.scatter(X[:, 0], X[:, 1])
        ax.scatter(centroids[:, 0], centroids[:, 1], marker="*")
        plt.show()

        _, ax = plt.subplots(figsize=(6, 6), dpi= 90)
        for l in labels:
            l = np.array(l)
            ax.scatter(X[l, 0], X[l, 1])
        plt.show()
