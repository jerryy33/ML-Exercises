# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def load_data(name: str, m=None, rng=None) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(name)
    x = data[:, :-1]
    y = data[:, -1]

    if not m is None:
        if rng is None:
            rng = np.default_rng(seed=66)
        idx = rng.choice(m, size=len(x), replace=False)
        x = x[idx]
        y = y[idx]

    return (x, y)


def plot(x: np.ndarray, y: np.ndarray, w: np.ndarray = None, sigma=None) -> None:
    """
    only for plotting 2D data
    """

    plt.plot(x, y, ".r", markersize=8, label="Samples")

    # also plot the prediction
    if not w is None:
        deg = w.shape[0]
        x_plot = np.linspace(np.min(x), np.max(x), 100)
        X_plot = np.vander(x_plot, deg)

        # set plotting range properly
        plt.ylim((np.min(y) * 1.2, np.max(y) * 1.2))

        plt.plot(
            x_plot, np.dot(X_plot, w), linewidth=5, color="tab:blue", label="Model"
        )

        # also plot confidence intervall
        if not sigma is None:
            plt.plot(x_plot, np.dot(X_plot, w) + sigma, linewidth=2, color="tab:cyan")
            plt.plot(x_plot, np.dot(X_plot, w) - sigma, linewidth=2, color="tab:cyan")

    plt.tight_layout()
    plt.savefig("fig.pdf")

    plt.show()


# Solves the linear least sqaures regression problem
def least_squares_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.column_stack((X, np.ones(len(X))))
    first = np.linalg.pinv(X.T @ X)
    return first @ X.T @ y


if __name__ == "__main__":
    x, y = load_data("dataset0.npy")
    w = least_squares_regression(x, y)
    plot(x, y, w)

    x, y = load_data("dataset1.npy")
    w = least_squares_regression(x, y)
    # plot(x, y, w)

    x, y = load_data("dataset2.npy")
    w = least_squares_regression(x, y)
    # plot(x, y, w)

    x, y = load_data("dataset3.npy")
    w = least_squares_regression(x, y)
    # plot(x, y, w)

    x, y = load_data("dataset4.npy")
    w = least_squares_regression(x, y)
    # plot(x, y, w)

    # print(np.linalg.norm(x @ w[:-1] + w[-1] - y))
