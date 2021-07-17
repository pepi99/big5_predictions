import sys
sys.path.append('../models')

from models.base_pipeline import BasePipeline
from models.arg_parser import create_parser
# from models.bert_model import BertModel
# from models.catboost_regr import CatboostRegr
# from models.neuralnet_multi import NeuralNetMulti
from models.bertweet import BertWrapper
from models.data_loader import DataLoader
#from models.bert_data_loader import DataLoader
from sklearn.neighbors import KNeighborsRegressor
from models.tfidf_model import TfidfModel
from models.scores import rmse
from models.scores import N_distance
from models.scores import percentage
from models.scores import idxs
from models.scores import average_words
from models.scores import lengths

from sklearn.model_selection import train_test_split
import numpy as np
import wandb


# python3 base_pipeline_example.py --train --use_bpe --data_path data/300_inf_full_en.csv
def main(args):
    wandb.init(project='big5', entity='zemerov')
    wandb.config.batch_size = args.batch_size

    dl = DataLoader()
    print('Loading data...')
    X, y = dl.parse_input(args.data_path, clean_text=args.clean_text)
    print('Lengths are: ', lengths(X))
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    y_train = y_train / 100

    if args.use_knn:
        print("Using KNN")
        embedder = TfidfModel()
        regressor = KNeighborsRegressor(n_jobs=-1)
    else:
        embedder = None
        regressor = BertWrapper(batch_size=args.batch_size, num_epochs=args.epochs, args=args)

    base_model = BasePipeline(embedder, regressor)

    if args.train:
        base_model.fit(X_train, y_train)
    else:
        base_model.load(args.model_path)

    y_pred = base_model.predict(X_test)
    y_pred = (y_pred * 100).astype(int)
    ten_dist = N_distance(y_test, y_pred, 10)
    five_dist = N_distance(y_test, y_pred, 5)

    _rmse = rmse(y_test, y_pred)
    indices = idxs(y_test, y_pred, 10)
    all_indices = [j for j in range(0, len(X_test))]

    print('Results: ')
    print('Shape of prediction set size: ', y_test.shape)
    print('10_distance: ', ten_dist)
    print('10_distance%: ', percentage(ten_dist, y_test.shape[0]))
    wandb.log({"10 distance": percentage(ten_dist, y_test.shape[0])})
    print('5_distance: ', five_dist)
    print('5_distance%: ', percentage(five_dist, y_test.shape[0]))
    print('rmse: ', _rmse)
    print('average words in error: ', average_words(X_test, indices))
    #print('Average words in correctly classified texts: ', average_words(X_test, list(set(all_indices) - set(indices))))
    #dl.connector.insert(X_test, y_test, y_pred)


if __name__ == '__main__':
    parser = create_parser()
    arguments = parser.parse_args()

    main(arguments)
