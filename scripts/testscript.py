from sklearn.metrics import mean_squared_error
import pickle
import pandas as pd
import re
from langdetect import detect
from tqdm import tqdm
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import tensorflow.keras as keras
def N_distance(y_true, y_pred, N):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    score = 0
    for y_i, yhat_i in zip(y_true, y_pred):
        vals = abs(y_i - yhat_i)
        if all(a <= N for a in vals):
            score += 1
    return score


def rmse(y, yhat):
    """

    :param y: actual M x N matrix
    :param yhat: predictions M x N matrix
    :return: rmse
    """
    return mean_squared_error(y, yhat, squared=False)


def idxs(y, yhat, N):
    """

    :param y: actual M x N matrix
    :param yhat: predictions M x N matrix
    :return:
    """
    indices = []
    for i, (y_i, yhat_i) in enumerate(zip(y, yhat)):
        vals = abs(y_i - yhat_i)
        if not (all(a <= N for a in vals)):  # If there is a problematic one
            indices.append(i)
    return indices


def average_words(texts, indices):
    av = 0
    for i in indices:
        av += len(texts[i].split())
    av /= len(indices)
    return av


def lengths(X):
    d = {100: 0, 1000: 0, 5000: 0, 10000: 0, 20000: 0}
    for text in X:
        l = len(text.split())
        print(l)
        for k in d.keys():
            if l >= k:
                d[k] += 1
            else:
                break
    return d


def percentage(x, y):
    """

    :param x: number
    :param y: number
    :return: x is what % of y?
    """
    return round((x * 100) / y, 2)

class DataLoader:
    def __init__(self):
        self.connector = None
    def detect_bad(self, text):
        lan = None
        try:
            lan = detect(text)
        except:
            lan = 'err'
            print('Found a bad text...')
        return lan

    def parse_input(self):
        # df = pd.read_csv(self.input_file)
        db_query = '''SELECT big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, input_text  FROM data_personality_analiser_nlp where input_text IS NOT NULL and input_text <> '' and input_text not like '%𝐇𝐞’𝐬 𝐜𝐮𝐦𝐦𝐢𝐧𝐠 𝐟𝐨𝐫 𝐛𝐞𝐬𝐭 ‘𝐃𝐢𝐜𝐤 𝐓𝐚𝐤𝐞𝐫’ 𝟐𝟎𝟐𝟏%' LIMIT 50000 '''
        #df = self.connector.query(db_query)
        #df = pd.read_csv('../data/10K_en_full.csv')
        #print('Shape of non-filtered df: ', df.shape)
        #print('Filtering out nonenglish texts...')

        #df = df[df.input_text.apply(self.detect_bad).eq('en')]
        #df = df[df.input_text.apply(lambda x: len(x.split()) in range(5000, 10000))]
        #print('Non english texts filtered!')
        #print('Shape of the DF: ', df.shape)
        #df.to_csv('../data/5K_10K_en.csv')
        #print('df saved!')
        df = pd.read_csv('../data/10K_en_full.csv')[:200]
        #df = pd.read_csv('../data/10K_en_full.csv')[:15000]
        y = df[['big5_openness', 'big5_conscientiousness', 'big5_extraversion', 'big5_agreeableness',
                'big5_neuroticism']].to_numpy()

        input_texts = df[['input_text']].astype(str).to_numpy()
        X = [re.sub(r'http\S+', '', text[0]) for text in input_texts]  # remove links
        X = self.tokenize_text(X)
        return X, y
    def tokenize_item(self, item):
        tokens = word_tokenize(item)
        stems = []
        for token in tokens:
            stems.append(PorterStemmer().stem(token))
        return stems
    def tokenize_text(self, text):
        res = []
        for txt in tqdm(text):
            res.append(' '.join(self.tokenize_item(txt.lower())))
        return res
nn = keras.models.load_model(
    '../cache/tfidf_pca_nn_full/keras-sequential')
pca = pickle.load(
    open('../cache/tfidf_pca_nn_full/pca_model',
         'rb'))
tfidf = pickle.load(open(
    '../cache/tfidf_pca_nn_full/tfidf_vectorizer',
    'rb'))
dl = DataLoader()
X, y = dl.parse_input()
mat = tfidf.transform(X).toarray()
reduced_mat = pca.transform(mat)
y_pred = nn.predict(reduced_mat)
ten_dist = N_distance(y, y_pred, 10)
five_dist = N_distance(y, y_pred, 5)
_rmse = rmse(y, y_pred)
indices = idxs(y, y_pred, 10)
all_indices = [j for j in range(0, len(X))]
print('Results: ')
print('Shape of prediction set size: ', y.shape)
print('10_distance: ', ten_dist)
print('10_distance%: ', percentage(ten_dist, y.shape[0]))
print('5_distance: ', five_dist)
print('5_distance%: ', percentage(five_dist, y.shape[0]))
print('rmse: ', _rmse)
print('average words in error: ', average_words(X, indices))
print('Average words in correctly classified texts: ', average_words(X, list(set(all_indices) - set(indices))))

