import os
from typing import List


class Basepipeline:
    def __init__(self, Embedder, Regressor):
        self.embedder = Embedder()
        self.regressor = Regressor()

    def fit(self, X: List[str], y: List[float]):
        self.embedder.fit(X)
        X_emb = self.embedder.encode(X)
        self.regressor.fit(X_emb, y)

    def predict(self, X: List[str]):
        X_emb = self.embedder.encode(X)
        return self.regressor.predict(X_emb)

    def save(self, odir):
        os.makedirs(odir, exist_ok=True)
        self.embedder.save(f'{odir}/{self.embedder.name}')
        self.regressor.save(f'{odir}/{self.regressor.name}')
