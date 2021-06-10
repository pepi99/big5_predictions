import os
from typing import List
import numpy as np


class Basepipeline:
    def __init__(self, Embedder, Regressor):
        self.embedder = Embedder()
        self.regressor = Regressor()

    def fit(self, X: List[str], y: List[float]):
        X_emb = self.embedder.fit(X)
        #4 X_emb = self.embedder.encode(X)
        self.regressor.fit(X_emb, y)

    def predict(self, X: List[str]):
        X_emb = self.embedder.encode(X)
        return self.regressor.predict(X_emb)

    def save(self, odir):
        os.makedirs(odir, exist_ok=True)
        self.embedder.save(f'{odir}/{self.embedder.name}')
        self.regressor.save(f'{odir}/{self.regressor.name}')
    def load(self, odir):
        self.embedder.load(odir)
        self.regressor.load(odir)
