from sklearn.metrics import mean_squared_error
from tensorflow.keras import backend as K
import tensorflow as tf


def N_distance(y_true, y_pred, N):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    y_r = tf.cast(K.round(y_true), dtype=tf.int64)
    ypred_r = tf.cast(K.round(y_pred), dtype=tf.int64)

    score = 0
    for y_i, yhat_i in zip(y_r, ypred_r):
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
def percentage(x, y):
    """
    
    :param x: number
    :param y: number
    :return: x is what % of y?
    """
    return round((x*100)/y, 2)

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
        for k in d.keys():
            if l >= k:
                d[k] += 1
            else:
                break
    return d

