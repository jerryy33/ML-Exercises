import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import DecisionBoundaryDisplay


def load_data(name, m=None):
    """Load .npy file and split into x,y data"""
    data = np.load(f"datasets/{name}")
    x = data[:, :-1]
    y = data[:, -1]

    return (x, y)


def grid_search(estimator: SVC, parameters: dict, x, y) -> tuple[SVC, float, dict]:
    """Perform grid search for a set of parameters and a model.
    Return the best estimator the best score and the best para set.
    """
    model = GridSearchCV(estimator, parameters)
    model.fit(x, y)
    return model.best_estimator_, model.best_score_, model.best_params_


def create_svc_with_param_set() -> tuple[SVC, dict]:
    """Create a svm for classification and provides a set of params to be used for tuning hyperparams"""
    parameters = {
        "kernel": ("linear", "poly", "rbf", "sigmoid"),
        "C": np.linspace(0.01, 10, 200),
    }
    svc = SVC()
    return svc, parameters


def determine_best_estimator(start_estimator, search_paramters) -> SVC:
    """Determines a "best" estimator via grid search and returns the one with best training score"""
    strongest_estimator, best_train_score, best_params = grid_search(
        start_estimator, search_paramters, x_train, y_train
    )
    print(f"Best parameters: {best_params}")
    print(f"Best train error: {best_train_score}")
    print(f"Best test error: {strongest_estimator.score(x_test, y_test)}", end="\n\n")
    return strongest_estimator


def plot(estimator: SVC, x_data, y_data) -> None:
    """Plot two dimensional data and the decision boundary of the estimator"""
    DecisionBoundaryDisplay.from_estimator(estimator, x_data, alpha=0.8, eps=0.5)
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data, edgecolors="k")
    plt.show()


if __name__ == "__main__":
    # create a base estimator(SVC) and and range of possible hyperparameters
    base, base_params = create_svc_with_param_set()

    print("Dataset O")
    x_train, y_train = load_data("dataset_O_train.npy")
    x_test, y_test = load_data("dataset_O_test.npy")
    best_estimator = determine_best_estimator(base, base_params)
    plot(best_estimator, x_train, y_train)

    print("Dataset U")
    x_train, y_train = load_data("dataset_U_train.npy")
    x_test, y_test = load_data("dataset_U_test.npy")
    best_estimator = determine_best_estimator(base, base_params)
    plot(best_estimator, x_train, y_train)

    print("Dataset V")
    x_train, y_train = load_data("dataset_V_train.npy")
    x_test, y_test = load_data("dataset_V_test.npy")
    best_estimator = determine_best_estimator(base, base_params)
