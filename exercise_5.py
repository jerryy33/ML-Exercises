from sklearn.linear_model import (
    Ridge,
    LinearRegression,
    Lasso,
    ElasticNet,
    TweedieRegressor,
)
from sklearn.linear_model._least_angle import Lars
from regression import load_data


if __name__ == "__main__":
    x, y = load_data("dataset_sparse_train.npy")
    x_test, y_test = load_data("dataset_sparse_test.npy")

    linar_regr = LinearRegression()
    linar_regr.fit(x, y)
    score1 = linar_regr.score(x_test, y_test)

    ridge = Ridge(alpha=0.5)
    ridge.fit(x, y)
    score2 = ridge.score(x_test, y_test)

    lasso = Lasso(alpha=0.5)
    lasso.fit(x, y)
    score3 = lasso.score(x_test, y_test)

    elastic_net = ElasticNet(alpha=0.1)
    elastic_net.fit(x, y)
    score4 = elastic_net.score(x_test, y_test)

    lars = Lars()
    lars.fit(x, y)
    score5 = lars.score(x_test, y_test)

    tweedie = TweedieRegressor(power=2, link="log")
    tweedie.fit(x, y)
    score6 = tweedie.score(x_test, y_test)

    # Elastic Net performs by far the best
    print(score1, score2, score3, score4, score5, score6)
