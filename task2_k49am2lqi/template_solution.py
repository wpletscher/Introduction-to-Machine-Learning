# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel, ExpSineSquared, Matern, RationalQuadratic
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training and test data
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    train_df = train_df.drop(columns=['season'])
    test_df = test_df.drop(columns=['season'])
    original_train_columns = train_df.columns
    original_test_columns = test_df.columns
    
    # Remove rows with NaN values for price_CHF in training data
    train_mask = train_df['price_CHF'].notna()
    train_df = train_df[train_mask]
    
    # Impute missing values in training and test data
    imputer = IterativeImputer(max_iter=10, random_state=42)
    train_df = imputer.fit_transform(train_df)
    test_df = imputer.fit_transform(test_df)
    train_df = pd.DataFrame(train_df, columns=original_train_columns)
    test_df = pd.DataFrame(test_df, columns=original_test_columns)

    X_train = train_df.drop(columns=['price_CHF'])
    y_train = train_df['price_CHF']
    X_test = test_df

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test



class Model(object):
    def __init__(self):
        super().__init__()
        self._x_train = None
        self._y_train = None
        self._kernel = RationalQuadratic()
        self._model = GaussianProcessRegressor(kernel=self._kernel)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self._x_train = X_train
        self._y_train = y_train
        self._model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred = self._model.predict(X_test)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred



# Main function. You don't have to change this
if __name__ == "__main__":
    X_train, y_train, X_test = data_loading()
    model = Model()
    model.train(X_train=X_train, y_train=y_train)
    y_pred = model.predict(X_test)
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

