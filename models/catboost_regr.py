import catboost

from models.regressor import Regressor


class CatboostRegr(Regressor):
    def __init__(self, *args, **kwargs):
        self.name = 'catboost'
        self.model = catboost.CatBoostRegressor(*args, **kwargs, objective='MultiRMSE', iterations=2000)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model = catboost.CatBoostRegressor()
        self.model.load_model(path)
