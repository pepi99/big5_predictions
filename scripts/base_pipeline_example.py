from models.base_pipeline import Basepipeline
from models.bert_model import BertModel
from models.catboost_regr import CatboostRegr
from models.neuralnet_multi import NeuralNetMulti
from models.data_loader import DataLoader
from models.tfidf_model import TfidfModel
from models.scores import rmse
from models.scores import N_distance


from sklearn.model_selection import train_test_split
import numpy as np


def main():
    dl = DataLoader()

    X, y = dl.parse_input()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    base_model = Basepipeline(TfidfModel, NeuralNetMulti)
    base_model.fit(X_train, y_train)
    base_model.save('../cache/tfidf_pca_nn_full/')
    y_pred = base_model.predict(X_test)

    ndist = N_distance(y_test, y_pred, 10)
    _rmse = rmse(y_test, y_pred)
    print('Results: ')
    print('Shape of prediction set size: ', y_test.shape)
    print('10_distance: ', ndist)
    print('rmse: ', _rmse)


if __name__ == '__main__':
    main()
