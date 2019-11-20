import numpy as np


def compute_a(x, x_lab, X, labels):
    index_points_in_cluster = np.where(labels == x_lab)
    points_in_cluster = X[index_points_in_cluster]

    norm = np.linalg.norm(x - points_in_cluster, axis=1)
    return np.sum(norm) / (norm.size-1)


def compute_b(x, x_lab, X, labels):
    bs = []
    other_clusters = np.unique(labels[labels != x_lab])

    for c in other_clusters:
        ind_points = np.where(labels == c)
        points = X[ind_points]
        bs.append(np.mean(np.linalg.norm(x - points, axis=1)))

    return np.min(bs)


def silhouette_samples(X, labels):
    """Evaluate the silhouette for each point and return them as a list.
    :param X: input data points, array, shape = (N,C).
    :param labels: the list of cluster labels, shape = N.
    :return: silhouette : array, shape = N
    """

    silhouettes = np.zeros(labels.size)
    for i, (x, lab) in enumerate(zip(X, labels)):
        a = compute_a(x, lab, X, labels)
        b = compute_b(x, lab, X, labels)
        silhouettes[i] = (b-a) / np.max([a, b])

    return silhouettes


def silhouette_score(X, labels):
    """Evaluate the silhouette for each point and return the mean.
    :param X: input data points, array, shape = (N,C).
    :param labels: the list of cluster labels, shape = N.
    :return: silhouette : float
    """
    return np.mean(silhouette_samples(X, labels))
