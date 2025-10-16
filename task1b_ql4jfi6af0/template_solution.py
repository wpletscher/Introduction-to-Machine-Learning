# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...

def transform_data(X):
    X_input = np.zeros((700, 21))

    X_input[:, 0:5] = X
    X_input[:, 5:10] = X ** 2
    X_input[:, 10:15] = np.exp(X)
    X_input[:, 15:20] = np.cos(X)
    X_input[:, 20] = 1

    assert X_input.shape == (700, 21)
    return X_input



def fit(X, y):
    weights = np.zeros((21,))
    X_input = transform_data(X)
    
    # Compute the optimal weights
    model = Ridge(alpha=314, fit_intercept=False, solver='lsqr')
    model.fit(X_input, y)
    weights = model.coef_
    
    assert weights.shape == (21,)
    return weights



# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
