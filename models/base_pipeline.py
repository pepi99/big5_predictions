import os
from typing import List
import numpy as np
from sklearn.preprocessing import StandardScaler


class BasePipeline:
    def __init__(self, embedder, regressor, normalize=False):
        self.embedder = embedder
        self.regressor = regressor
        self.X_train_emb = None
        self.normalize = normalize

        self.scaler = StandardScaler()

    def fit(self, X: List[str], y: List[float]):
        if self.embedder is not None:
            self.X_train_emb = self.embedder.fit(X)
        else:
            self.X_train_emb = X

        if self.normalize:
            print("Normalization")
            self.X_train_emb = self.scaler.fit_transform(X=self.X_train_emb, y=None)

        self.regressor.fit(self.X_train_emb, y)

    def predict(self, X: List[str]):
        if self.embedder is not None:
            X_emb = self.embedder.encode(X)
        else:
            X_emb = X

        if self.normalize:
            print("Normalization")
            X_emb = self.scaler.transform(X_emb)

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
