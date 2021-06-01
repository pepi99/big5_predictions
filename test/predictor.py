import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
import sys

np.set_printoptions(threshold=sys.maxsize)


def error_function(x1, x2):
    error = 0
    for v1, v2 in zip(x1, x2):
        error += abs(v1 - v2)
    error = error / len(x1)
    return error


def N_distance(x1, x2, N):
    distance = 0
    for v1, v2 in zip(x1, x2):
        if abs(v1 - v2) <= N:
            distance += 1
    return distance


X = np.loadtxt('../input_data/X_full.txt')
y = np.loadtxt('../input_data/Y_full.txt')
y = y[:, 1]  # Take first feature only

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.1, random_state=1)

print(X_train.shape)
print(y_train.shape)
print(X_validate.shape)
print(y_validate.shape)

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

yhat = model.predict(X_validate)
yhat = np.rint(yhat)  # Round

error = MSE(y_validate, yhat, squared=False)
distance = N_distance(y_validate, yhat, 10)
print('N distance for N=10 is: ', distance)
print('Size of validation set: ', y_validate.shape)
print('Error is: ', error)
