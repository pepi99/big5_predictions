from abc import ABC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
from models.embedder import Embedder
import pickle


# nltk.download('punkt')


class TfidfModel(Embedder, ABC):
    """
    Tf-idf + PCA.
    """

    def __init__(self):
        self.name = ''
        self.model = TfidfVectorizer(lowercase=True, max_features=50000)
        self.pca = PCA(n_components=100)

    def fit(self, X):
        print('Fitting the tfidf vectorizer...')
        tokenized_text = self.tokenize_text(X)
        matrix = self.model.fit_transform(tokenized_text).todense()
        matrix = np.squeeze(np.asarray(matrix))
        print('Dimension of original tfidf matrix: ', matrix.shape)

        self.pca.fit(matrix)
        reduced_matrix = self.pca.transform(matrix)
        print('Dimension of reduced matrix: ', reduced_matrix.shape)
        print('Encoder fitting completed!')
        return reduced_matrix

    def encode(self, X):
        print('Encoding data...')
        tokenized_text = self.tokenize_text(X)
        matrix = self.model.transform(tokenized_text).todense()
        matrix = np.squeeze(np.asarray(matrix))
        reduced_matrix = self.pca.transform(matrix)
        return reduced_matrix

    def load(self, path):
        pass

    def save(self, odir):
        print('Saving model...')
        pickle.dump(self.model, open(odir + 'tfidf_vectorizer', 'wb'))  # Save tfidf vectorizer
        pickle.dump(self.pca, open(odir + 'pca_model', 'wb'))  # Save the PCA model
        print('Model saved!')

    def tokenize_item(self, item):
        tokens = word_tokenize(item)
        stems = []
        for token in tokens:
            stems.append(PorterStemmer().stem(token))
        return stems

    def tokenize_text(self, text):
        return [' '.join(self.tokenize_item(txt.lower())) for txt in text]
