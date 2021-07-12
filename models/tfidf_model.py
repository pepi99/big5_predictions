from abc import ABC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import numpy as np
from .embedder import Embedder
from .tokenizer_wrapper import TokenizerWrapper

import pickle
from nltk import download
import transformers as t
download('punkt')


class TfidfModel(Embedder, ABC):
    """
    Tf-idf
    """

    def __init__(self, use_bpe=False):
        self.name = ''
        self.use_bpe = use_bpe

        if use_bpe:
            hf_tokenizer = t.AutoTokenizer.from_pretrained('roberta-base', use_fast=True)
            self.model = TfidfVectorizer(
                lowercase=True,
                max_features=20000,
                tokenizer=TokenizerWrapper(hf_tokenizer)
            )
        else:
            self.model = TfidfVectorizer(lowercase=True, max_features=20000)

    def fit(self, X):
        print('Fitting the tfidf vectorizer...')

        matrix = self.model.fit_transform(X).todense()

        print('Fitting the tfidf vectorizer finished!')

        matrix = np.squeeze(np.asarray(matrix))

        print('Dimension of original tfidf matrix: ', matrix.shape)

        reduced_matrix = matrix

        print('Encoder fitting completed!')

        return reduced_matrix

    def encode(self, X):
        print('TfIdf transforming test data...')
        matrix = self.model.transform(X).todense()
        print('TfIdf transform finished!')
        matrix = np.squeeze(np.asarray(matrix))
        print('PCA transformign finsihed!')

        return matrix

    def load(self, path):
        print('Loading pca and tfidf models...')

        self.model = pickle.load(open(path + '/tfidf_vectorizer', 'rb'))

        print('Models loaded!')

    def save(self, odir):
        print('Saving model...')
        pickle.dump(self.model, open(odir + 'tfidf_vectorizer', 'wb'))  # Save tfidf vectorizer
        print('Model saved!')
