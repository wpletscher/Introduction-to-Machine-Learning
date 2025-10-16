import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

Id_train = train_data['Id']
y_train = train_data['y']
X_train = train_data.drop(columns=['Id','y'])

Id_test = test_data['Id']
X_test = test_data.drop(columns=['Id'])


model = LinearRegression()
# model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# mse = mean_squared_error(y, y_pred)

output_data = pd.DataFrame({'Id': test_data['Id'], 'y': y_pred})
output_data.to_csv('predictions.csv', index=False)

# print(mse)