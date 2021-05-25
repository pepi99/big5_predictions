import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

X = np.loadtxt('../input_data/X.txt')[:10]
y = np.loadtxt('../input_data/Y.txt')[:10]
y = y[:, 0]
print(X.shape)
print(y.shape)
model = xgb.XGBRegressor()

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )