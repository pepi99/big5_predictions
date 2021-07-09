from abc import ABC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
from embedder import Embedder
import pickle
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk import download
download('punkt')


class TfidfModel(Embedder, ABC):
    """
    Tf-idf + PCA.
    """

    def __init__(self):
        self.name = ''
        self.model = TfidfVectorizer(lowercase=True, max_features=20000)
        self.pca = PCA(n_components=9000)

    def fit(self, X):
        #print('Tokenizing training data...')
        #tokenized_text = self.tokenize_text(X)
        #print('Tokenizing training data finished!')
        print('Fitting the tfidf vectorizer...')
        matrix = self.model.fit_transform(X).todense()
        print('Fitting the tfidf vectorizer finished!')
        matrix = np.squeeze(np.asarray(matrix))
        print('Dimension of original tfidf matrix: ', matrix.shape)
        
        print('Fit transforming the PCA on the training data...')
        self.pca.fit(matrix)
        reduced_matrix = self.pca.transform(matrix)
        print('Fit transforming of the PCA training data finished!')
        print('Dimension of reduced matrix: ', reduced_matrix.shape)
        print('Encoder fitting completed!')
        return reduced_matrix

    def encode(self, X):
        #print('Tokenizing test data...')
        #tokenized_text = self.tokenize_text(X)
        #print('Tokenizing test data finished!')
        print('TfIdf transforming test data...')
        matrix = self.model.transform(X).todense()
        print('TfIdf transform finished!')
        matrix = np.squeeze(np.asarray(matrix))
        print('PCA transforming test data...')
        reduced_matrix = self.pca.transform(matrix)
        print('PCA transformign finsihed!')
        return reduced_matrix

    def load(self, path):
        print('Loading pca and tfidf models...')
        self.model = pickle.load(open(path + '/tfidf_vectorizer', 'rb'))
        self.pca = pickle.load(open(path + '/pca_model', 'rb'))
        print('Models loaded!')
     

    def save(self, odir):
        print('Saving model...')
        pickle.dump(self.model, open(odir + 'tfidf_vectorizer', 'wb'))  # Save tfidf vectorizer
        pickle.dump(self.pca, open(odir + 'pca_model', 'wb'))  # Save the PCA model
        print('Model saved!')
