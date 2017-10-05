# -*- coding: utf-8 -*-

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


class Recognizer(object):
    class AccuracyLogger(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.acc = []

        def on_epoch_end(self, batch, logs=None):
            if logs:
                self.acc.append(logs.get('acc'))

    def __init__(self, batch_size=10, epochs=1, img_x=28, img_y=28):
        self._model = None

        self._num_classes = 10
        self._img_x = img_x
        self._img_y = img_y

        self._batch_size = batch_size
        self._epochs = epochs

        self._acc_logger = self.AccuracyLogger()

        self._train_data_X = None
        self._train_data_Y = None

        self._test_data_X = None
        self._test_data_Y = None

    def load_data(self, train_data, test_data):
        self._train_data_X, self._train_data_Y = train_data
        self._test_data_X, self._test_data_Y = test_data

        self._train_data_X = self._train_data_X.reshape(
            self._train_data_X.shape[0],
            self._img_x,
            self._img_y,
            1
        ).astype('float32')
        self._train_data_X /= 255

        self._test_data_X = self._test_data_X.reshape(
            self._test_data_X.shape[0],
            self._img_x,
            self._img_y,
            1
        ).astype('float32')
        self._test_data_X /= 255

        self._train_data_Y = keras.utils.to_categorical(self._train_data_Y, self._num_classes)
        self._test_data_Y = keras.utils.to_categorical(self._test_data_Y, self._num_classes)

    def build_model(self):
        self._model = Sequential()
        self._model.add(
            Conv2D(
                filters=32,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation='relu',
                input_shape=(self._img_x, self._img_y, 1)
            )
        )
        self._model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self._model.add(
            Conv2D(
                filters=64, kernel_size=(5, 5), activation='relu'
            )
        )
        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Flatten())

        self._model.add(Dense(units=1000, activation='relu'))
        self._model.add(Dense(units=self._num_classes, activation='softmax'))

        self._model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy']
        )

    def train_model(self, verbose=0):
        self._model.fit(
            self._train_data_X,
            self._train_data_Y,
            batch_size=self._batch_size,
            epochs=self._epochs,
            verbose=verbose,
            validation_data=(self._test_data_X, self._test_data_Y),
            callbacks=[self._acc_logger]
        )

    def evaluate_model(self):
        return self._model.evaluate(self._test_data_X, self._test_data_Y, verbose=0)[1]

    def save_model_weights(self, filename):
        self._model.save_weights(filename)

    def save_model(self, model_filename, weights_filename=None):
        with open(file=model_filename, mode='w', encoding='utf8') as outmodel:
            outmodel.write(self._model.to_json())

        if weights_filename:
            self._model.save_weights(weights_filename)
