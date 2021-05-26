import unittest

from models.catboost_regr import CatboostRegr


class TestRegr(unittest.TestCase):
    def test_catboost(self):
        X = [[1.1, 0, 0.1, 1, 1], [1.1, 0.5, 0.1, 1, 1], [1.1, 0.3, 0.1, 1, 1], [1.1, 0, 0.1, 1.4, 0.93]]
        y = [[10, 20, 30, 40, 50],
             [10, 22, 31, 40, 41],
             [90, 93, 95, 91, 98],
             [99, 100, 45, 11, 37]
             ]
        cb = CatboostRegr()
        cb.fit(X, y)
        y_pred = cb.predict(X)
        print(y_pred)
