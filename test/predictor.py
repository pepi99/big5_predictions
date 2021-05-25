import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error


def error_function(x1, x2):
    error = 0
    for v1, v2 in zip(x1, x2):
        error += abs(v1 - v2)
    error = error / len(x1)
    return error


X = np.loadtxt('../input_data/X.txt')
y = np.loadtxt('../input_data/Y.txt')

y = y[:, 0]

print(X.shape)
print(y.shape)
model = xgb.XGBRegressor()
model.fit(X[:9000], y[:9000])
yhat = model.predict(X[9000:])

error = error_function(y[9000:], yhat)
print('Error is: ', error)
# print(yhat)
# print(yhat)
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# scores = np.absolute(scores)
# print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))
# yhat = model.
