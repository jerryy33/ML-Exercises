"""Gradient descent from scratch with backtracking line search"""
from typing import Callable
import numpy as np
from regression import load_data, least_squares_regression, plot


def backtracking_line_search(
    f: callable,
    X: np.ndarray,
    Y: np.ndarray,
    w: np.ndarray,
    delta_x: np.ndarray,
    a: float = 0.3,  # Between 0 and 1
    b: float = 0.8,  # Between 0 and 1
) -> float:
    # t is step length parameter
    t = 1
    while f(X, Y, w + t * delta_x) > f(X, Y, w) + (
        a * t * grad_least_squares(X, Y, w).T @ delta_x
    ):
        t *= b
    return t


def gradient_descent(
    f: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    # start at arbitrary point w based on x's dimension
    w = np.array(np.reshape(X[0], (X.shape[1], 1)), dtype="float64")

    # Calculate gradient and reverse it to walk to the next minimum
    delta_x = -grad_least_squares(X, Y, w)

    # Loop as long as gradient changes more than threshold
    while np.linalg.norm(delta_x) > 0.000001:
        # print(np.linalg.norm(delta_x))
        # backtacking line search to get best step rate
        step_rate = backtracking_line_search(f, X, Y, w, delta_x)

        # update starting point
        w += step_rate * delta_x

        # update gradient at the new point
        delta_x = -grad_least_squares(X, Y, w)
    return w


# RSS loss function
def loss(x: np.ndarray, y: np.ndarray, b: np.ndarray) -> np.ndarray:
    Xb = x @ b
    y_minus_xb = y[:, None] - Xb
    return y_minus_xb.T @ y_minus_xb
    # alternative definition: return np.sum((y - (x.T * b)) ** 2)


# Gradinet of RSS loss function
def grad_least_squares(x: np.ndarray, y: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2 * x.T @ (x @ b - y[:, None])


# Normal equation for comparing results
def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    first = np.linalg.pinv(X.T @ X)
    return first @ X.T @ y


if __name__ == "__main__":
    # load our dataset
    x, y = load_data("dataset4.npy")
    x_extended = np.column_stack((x, np.ones(len(x))))

    # Descent to global minumum
    d = gradient_descent(loss, x_extended, y).flatten()

    # Compute RSS
    pred = x_extended @ d
    e = y - pred
    rss = np.linalg.norm(e, ord=2)

    # Results

    # Solves normal equation for comparison
    normal_solution = normal_equation(x_extended, y)

    # least squares solution to compare actual coefficients
    w = least_squares_regression(x, y)
    print(
        f"Correct coefficients: {w}\nGradient solution: {d}\nNormal equation: {normal_solution}\nRSS: {rss}"
    )
    plot(x, y, d)
