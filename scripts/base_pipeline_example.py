from models.base_pipeline import Basepipeline
from models.bert_model import BertModel
from models.catboost_regr import CatboostRegr


def main():
    X = ['abc', 'abcd.', 'abcd efgh', 'abcd']

    y = [[10, 20, 30, 40, 50],
         [10, 22, 31, 40, 41],
         [90, 93, 95, 91, 98],
         [99, 100, 45, 11, 37]
         ]

    base_model = Basepipeline(BertModel, CatboostRegr)
    base_model.fit(X, y)
    base_model.save('../cache/base_model')
    y_pred = base_model.predict(X)
    print(y_pred)


if __name__ == '__main__':
    main()
