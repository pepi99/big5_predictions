import os
from typing import List
import numpy as np


class BasePipeline:
    def __init__(self, embedder, regressor):
        self.embedder = embedder
        self.regressor = regressor

    def fit(self, X: List[str], y: List[float]):
        if self.embedder is not None:
            X_emb = self.embedder.fit(X)
        else:
            X_emb = X

        self.regressor.fit(X_emb, y)

    def predict(self, X: List[str]):
        if self.embedder is not None:
            X_emb = self.embedder.encode(X)
        else:
            X_emb = X

        return self.regressor.predict(X_emb)

    def save(self, odir):
        os.makedirs(odir, exist_ok=True)

        if self.embedder is not None:
            self.embedder.save(f'{odir}/{self.embedder.name}')

        self.regressor.save(f'{odir}/{self.regressor.name}')

    def load(self, odir):
        if self.embedder is not None:
            self.embedder.load(odir)

        self.regressor.load(odir)
