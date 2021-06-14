from regressor import Regressor
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow.keras
from tensorflow.keras import callbacks

import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices())


class NeuralNetMulti(Regressor):
    def __init__(self):
        self.name = 'keras-sequential'
        self.model = Sequential()
        # self.earlystopping = callbacks.EarlyStopping(monitor="mae",
        #                                              mode="min", patience=5,
        #                                              restore_best_weights=True)

    def fit(self, X, y):
        print('Fitting into the neural net...')
        n_inputs = X.shape[1]
        n_outputs = y.shape[1]
        self.model.add(Dense(256, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
        #self.model.add(Dense(100, activation='relu'))
        #self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(n_outputs))
        self.model.summary()
        self.model.compile(loss='mae', optimizer='adam', metrics=['mse', 'mae'])
        self.model.fit(X, y, verbose=1, epochs=450)
        # self.model.fit(X, y, verbose=1, epochs=1000, callbacks=[self.earlystopping])
        print('Fitting completed!')

    def predict(self, X):
        print('Predicting...')
        predictions = self.model.predict(X, verbose=1)
        print('Predicted!')
        return predictions

    def save(self, path):
        print('Saving model to ', path, '...')
        self.model.save(path)
        print('Model saved')

    def load(self, path):
        print('Loading NN  model...')
        self.model = tf.keras.models.load_model(path + '/keras-sequential')
        print('NN Model loaded!')

    # def get_dataset(self):
    #     X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=3, random_state=2)
    #     return X, y

    # evaluate a model using repeated k-fold cross-validation
    # def evaluate_model(self, X, y):
    #     results = list()
    #     n_inputs, n_outputs = X.shape[1], y.shape[1]
    #     # define evaluation procedure
    #     cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    #     # enumerate folds
    #     for train_ix, test_ix in cv.split(X):
    #         # prepare data
    #         X_train, X_test = X[train_ix], X[test_ix]
    #         y_train, y_test = y[train_ix], y[test_ix]
    #         # define model
    #         model = get_model(n_inputs, n_outputs)
    #         # fit model
    #         model.fit(X_train, y_train, verbose=0, epochs=100)
    #         # evaluate model on test set
    #         mae = model.evaluate(X_test, y_test, verbose=0)
    #         # store result
    #         print('>%.3f' % mae)
    #         results.append(mae)
    #     return results
# READ THIIIIIIIIIIS
# https://stackoverflow.com/questions/56299770/units-in-dense-layer-in-keras/56302896
