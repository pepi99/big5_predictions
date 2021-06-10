from sklearn.metrics import mean_squared_error


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

