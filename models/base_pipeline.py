import os
from typing import List
import numpy as np


class BasePipeline:
    def __init__(self, embedder, regressor):
        self.embedder = embedder
        self.regressor = regressor

    def fit(self, X: List[str], y: List[float]):
        X_emb = self.embedder.fit(X)

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
