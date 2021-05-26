import abc


class Embedder(abc.ABC):
    @abc.abstractmethod
    def encode(self, X):
        pass

    @abc.abstractmethod
    def fit(self, X):
        pass
