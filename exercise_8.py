import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)


def load_data(name, m=None):
    data = np.load(f"datasets/exercise8/{name}")
    x = data[:, :-1]
    y = data[:, -1]

    return (x, y)


def plot_numbers(numb, tag, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed=66)
    ones = [numb[tag == i][0] for i in range(5)]
    twos = [numb[tag == (i + 5)][0] for i in range(5)]
    (fig, axs) = plt.subplots(nrows=2, ncols=5)
    for ax, i in zip(axs[0], range(5)):
        ax.imshow(ones[i].reshape(8, 8), cmap="gray", vmin=0, vmax=16)
    for ax, i in zip(axs[1], range(5)):
        ax.imshow(twos[i].reshape(8, 8), cmap="gray", vmin=0, vmax=16)
    plt.show()


def fit_knn(x_data, y_data, k=3) -> KNeighborsClassifier:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_data, y_data)
    return knn


def fit_lda(x_data, y_data) -> LinearDiscriminantAnalysis:
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_data, y_data)
    return lda


def fit_qda(x_data, y_data) -> QuadraticDiscriminantAnalysis:
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(x_data, y_data)
    return qda


def classifier_score(
    classifier: LinearDiscriminantAnalysis
    | KNeighborsClassifier
    | QuadraticDiscriminantAnalysis,
    x_data,
    y_data,
    error_set="test",
) -> None:
    score = classifier.score(x_data, y_data)
    print(f"Best {error_set} error for {classifier.__class__.__name__}", score)


def train_all_classifiers(
    x_data, y_data
) -> tuple[
    KNeighborsClassifier, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
]:
    knn = fit_knn(x_data, y_data)
    lda = fit_lda(x_data, y_data)
    qda = fit_qda(x_data, y_data)
    return knn, lda, qda


if __name__ == "__main__":
    # Exercise 1
    print("Dataset R")
    x_train, y_train = load_data("dataset_R_train.npy")
    x_test, y_test = load_data("dataset_R_test.npy")
    regressor = KNeighborsRegressor(n_neighbors=5)
    regressor.fit(x_train, y_train)
    regressor_score = regressor.score(x_test, y_test)

    print("Best test error for knn regression: ", regressor_score, end="\n\n")

    # Exercise 2-4
    print("\nDataset E")
    x_train, y_train = load_data("dataset_E_train.npy")
    x_test, y_test = load_data("dataset_E_test.npy")

    knn_classifier, lda_classifier, qda_classifier = train_all_classifiers(
        x_train, y_train
    )

    classifier_score(knn_classifier, x_test, y_test)
    classifier_score(lda_classifier, x_test, y_test)
    classifier_score(qda_classifier, x_test, y_test)

    print("\nDataset G")
    x_train, y_train = load_data("dataset_G_train.npy")
    x_test, y_test = load_data("dataset_G_test.npy")

    knn_classifier, lda_classifier, qda_classifier = train_all_classifiers(
        x_train, y_train
    )

    classifier_score(knn_classifier, x_test, y_test)
    classifier_score(lda_classifier, x_test, y_test)
    classifier_score(qda_classifier, x_test, y_test)

    print("\nDataset O")
    x_train, y_train = load_data("dataset_O_train.npy")
    x_test, y_test = load_data("dataset_O_test.npy")

    knn_classifier, lda_classifier, qda_classifier = train_all_classifiers(
        x_train, y_train
    )

    classifier_score(knn_classifier, x_test, y_test)
    classifier_score(lda_classifier, x_test, y_test)
    classifier_score(qda_classifier, x_test, y_test)

    # Exercise 5
    print("\n\nDataset digits")
    X, y = load_digits(return_X_y=True)
    plot_numbers(X, y)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=66
    )
    knn_classifier, lda_classifier, qda_classifier = train_all_classifiers(
        x_train, y_train
    )

    classifier_score(knn_classifier, x_train, y_train, error_set="train")
    classifier_score(lda_classifier, x_train, y_train, error_set="train")
    classifier_score(qda_classifier, x_train, y_train, error_set="train")

    classifier_score(knn_classifier, x_test, y_test)
    classifier_score(lda_classifier, x_test, y_test)
    classifier_score(qda_classifier, x_test, y_test)
