from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from regression import load_data


if __name__ == "__main__":
    x, y = load_data("dataset_numbers_train.npy")
    x_test, y_test = load_data("dataset_numbers_test.npy")

    svm = LinearSVC(penalty="l2", loss="squared_hinge", C=0.1)
    svm.fit(x, y)
    svm_score = svm.score(x_test, y_test)

    lg = LogisticRegression(penalty="l2", solver="liblinear", C=0.2)
    lg.fit(x, y)
    lg_score = lg.score(x_test, y_test)

    # Scores are mean accuracy
    print(lg_score, svm_score)
    print(f"Best test error: {max(lg_score, svm_score)}")
