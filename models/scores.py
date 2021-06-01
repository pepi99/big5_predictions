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
