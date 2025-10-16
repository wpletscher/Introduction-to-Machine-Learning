# This serves as a template which will guide you through the implementation of this task. It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error


# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...

def fit(X, y, lam):
    weights = np.zeros((13,))
    
    model = Ridge(alpha=lam, fit_intercept=True)
    model.fit(X, y)
    weights = model.coef_
    bias = model.intercept_

    #assert weights.shape == (13,)
    return weights , bias


def calculate_RMSE(w, b, X, y):
    rmse = 0
    y_hat = X @ w + b
    rmse = root_mean_squared_error(y_hat, y)
    assert np.isscalar(rmse)
    return rmse


def average_LR_RMSE(X, y, lambdas, n_folds):
    RMSE_mat = np.zeros((n_folds, len(lambdas)))

    kf = KFold(n_splits=n_folds,shuffle=True,random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for j, lam in enumerate(lambdas):
            w , b = fit(X_train, y_train, lam)
            RMSE_mat[i, j] = calculate_RMSE(w, b, X_test, y_test)


    avg_RMSE = np.mean(RMSE_mat, axis=0)
    assert avg_RMSE.shape == (5,)
    return avg_RMSE


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns="y")
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function calculating the average RMSE
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)
    # Save results in the required format
    np.savetxt("./results.csv", avg_RMSE, fmt="%.12f")
