from random import randrange
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from regression import load_data


def constrained_least_squares_regression(
    X: np.ndarray,
    y: np.ndarray,
    lamb: int = 0,
) -> np.ndarray:
    X = np.column_stack((X, np.ones(len(X))))
    first = np.linalg.pinv(X.T @ X + lamb * np.identity(X.shape[1]))
    return first @ X.T @ y


def polynomial_regression(
    X: np.ndarray, y: np.ndarray, degree: int = 2, lamb: float = 0
) -> np.ndarray:
    polynomial_features = PolynomialFeatures(degree=degree)
    X = polynomial_features.fit_transform(X)
    first = np.linalg.pinv(X.T @ X + lamb * np.identity(X.shape[1]))
    return first @ X.T @ y


def k_fold_cross_validation(
    x_data: np.ndarray,
    y_data: np.ndarray,
    degree: int,
    lambd: float,
    k: int = 10,
) -> tuple[float, float]:
    train_error = np.zeros((k, 1))
    val_error = np.zeros((k, 1))

    for i in range(0, k):
        window_size = x.shape[0] // k
        val_data = (
            x_data[i * window_size : i * window_size + window_size],
            y_data[i * window_size : i * window_size + window_size],
        )
        indexes = np.r_[0 : i * window_size, i * window_size + window_size : x.shape[0]]
        train_data = (
            x_data[indexes],
            y_data[indexes],
        )
        cofs = polynomial_regression(train_data[0], train_data[1], degree, lambd)
        train_result = error_func(train_data[0], train_data[1], cofs)
        val_result = error_func(val_data[0], val_data[1], cofs)
        train_error[i] = train_result
        val_error[i] = val_result

    return np.mean(train_error), np.mean(val_error)


def error_func(x, y, w):
    return 1 / (2 * x.shape[0]) * np.linalg.norm(np.polyval(w, x) - y, ord=2) ** 2


def lambda_selections():
    return [randrange(-1000000000, 1000) for _ in range(100)]


if __name__ == "__main__":
    x, y = load_data("dataset_poly_train.npy")
    x_test, y_test = load_data("dataset_poly_test.npy")

    # Exercise 2
    beta = constrained_least_squares_regression(x, y, lamb=2)
    print(beta)

    # Exercise 3
    pbeta = polynomial_regression(x, y, 6)
    print(pbeta)

    # Exercise 4 and 5
    best_lambd = 0
    val_errors = []
    train_errors = []
    lambds = lambda_selections()
    for lambd in lambds:
        train_error, validation_error = k_fold_cross_validation(
            x,
            y,
            6,
            lambd,
        )
        val_errors.append(validation_error)
        train_errors.append(train_error)
    plt.plot(np.arange(100), train_errors)
    plt.show()
    plt.plot(np.arange(100), val_errors)
    plt.show()

    # Exercise 6
    best_lambd = lambds[np.argmin(val_errors)]
    best_train_result = polynomial_regression(x, y, 6, best_lambd)
    best_test_result = polynomial_regression(x_test, y_test, 6, best_lambd)
    print(error_func(x, y, best_train_result))
    print(error_func(x_test, y_test, best_test_result))
