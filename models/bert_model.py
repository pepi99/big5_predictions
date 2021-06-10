from abc import ABC

from sentence_transformers import SentenceTransformer

from embedder import Embedder


class BertModel(Embedder, ABC):
    def __init__(self):
        self.name = 'bert'
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def encode(self, X):
        return self.model.encode(X)

    def fit(self, X):
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass
