import sys
sys.path.append('../models')
from base_pipeline import Basepipeline
#from bert_model import BertModel
from catboost_regr import CatboostRegr
from neuralnet_multi import NeuralNetMulti
from data_loader import DataLoader
#from bert_data_loader import DataLoader
from tfidf_model import TfidfModel
from scores import rmse
from scores import N_distance
from scores import percentage
from scores import idxs
from scores import average_words
from scores import lengths

from sklearn.model_selection import train_test_split
import numpy as np


def main():
    dl = DataLoader()

    X, y = dl.parse_input()
    print('Lengths are: ', lengths(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    y_train = y_train/100

    base_model = Basepipeline(TfidfModel, NeuralNetMulti)
    base_model.fit(X_train, y_train)
    base_model.save('../cache/tfidf_pca_nn_300_inf_full_en')
    #base_model.load('../cache/tfidf_pca_nn_full')
    y_pred = base_model.predict(X_test)
    y_pred = y_pred*100
    ten_dist = N_distance(y_test, y_pred, 10)
    five_dist = N_distance(y_test, y_pred, 5)
    _rmse = rmse(y_test, y_pred)
    indices = idxs(y_test, y_pred, 10)
    all_indices = [j for j in range(0, len(X_test))]
    print('Results: ')
    print('Shape of prediction set size: ', y_test.shape)
    print('10_distance: ', ten_dist)
    print('10_distance%: ', percentage(ten_dist, y_test.shape[0]))
    print('5_distance: ', five_dist)
    print('5_distance%: ', percentage(five_dist, y_test.shape[0]))
    print('rmse: ', _rmse)
    print('average words in error: ', average_words(X_test, indices))
    print('Average words in correctly classified texts: ', average_words(X_test, list(set(all_indices) - set(indices))))
    #dl.connector.insert(X_test, y_test, y_pred)


if __name__ == '__main__':
    main()
