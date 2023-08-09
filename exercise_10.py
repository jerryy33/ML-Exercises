import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA


def fit_and_plot(x, model: PCA | MDS) -> None:
    x = model.fit_transform(x)
    plt.figure().add_subplot(111, projection="3d").plot(*x.T, "o")
    plt.show()


if __name__ == "__main__":
    mds = MDS(metric=True, normalized_stress="auto")
    pca = PCA(n_components=2)

    X = np.load("./datasets/box.npy")
    fit_and_plot(X, mds)
    fit_and_plot(X, pca)

    X = np.load("./datasets/spring_1.npy")
    fit_and_plot(X, mds)
    fit_and_plot(X, pca)

    X = np.load("./datasets/spring_2.npy")
    fit_and_plot(X, mds)
    fit_and_plot(X, pca)
