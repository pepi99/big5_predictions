from abc import ABC
from embedder import Embedder


class BasicEmbedder(Embedder, ABC):

    def __init__(self):
        pass

    def fit(self, X):
        return X

    def encode(self, X):
        return X

    def load(self, path):
        pass

    def save(self, odir):
        pass

