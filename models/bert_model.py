from abc import ABC

from sentence_transformers import SentenceTransformer

from embedder import Embedder


class BertModel(Embedder, ABC):
    def __init__(self):
        self.name = 'bert'
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def encode(self, X):
        print('Encoding with Bert...')
        encoding = self.model.encode(X)
        print('Encoding with bert finished!')
        return encoding

    def fit(self, X):
        print('Encoding with Bert...')
        encoding = self.encode(X)
        print('Encoding with bert finished!')
        return encoding

    def load(self, path):
        pass

    def save(self, path):
        pass
