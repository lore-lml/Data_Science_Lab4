import numpy as np
import matplotlib.pyplot as plt
import script1

if __name__ == '__main__':
    labels = np.loadtxt("data_sets/kmean_result.txt", delimiter=",", skiprows=1, dtype=np.int_)
    X = script1.get_coordinates_from_file("data_sets/2d_gauss_clusters.txt")
    colors = np.random.rand(15)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=90)

    for i in range(15):
        mask = labels[:, 1] == i
        mask2 = labels[mask, 0]
        points = X[mask2]
        ax.scatter(points[:, 0], points[:, 1])
    plt.show()