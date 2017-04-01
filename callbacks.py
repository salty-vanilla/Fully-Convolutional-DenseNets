# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import os
import keras.callbacks
try:
    import cPickle as pickle
except:
    import pickle
import csv
from collections import deque
from collections import OrderedDict
from collections import Iterable


class BatchLogger(keras.callbacks.CSVLogger):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.on_epoch_end = keras.callbacks.Callback.on_epoch_end

        dst_dir = os.path.dirname(file_path)
        if dst_dir is not '':
            os.makedirs(dst_dir, exist_ok=True)

    def on_batch_end(self, batch, logs=None):
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['batch'] + self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'batch': batch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()


class ModelSaver(keras.callbacks.ModelCheckpoint):
    def __init__(self, file_path, verbose=0, save_freq=1):
        super().__init__(file_path, verbose=verbose)
        self.save_freq = save_freq

        dst_dir = os.path.dirname(file_path)
        if dst_dir is not '':
            os.makedirs(dst_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == 0:
            super().on_epoch_end(epoch, logs=logs)


# TODO segmentation用に作る
class Visualizer(keras.callbacks.Callback):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def on_epoch_end(self, epoch, logs=None):
        predict = self.model.predict(self.x)
        print(predict.shape)


def test():
    '''Trains a simple deep NN on the MNIST dataset.
    Gets to 98.40% test accuracy after 20 epochs
    (there is *a lot* of margin for parameter tuning).
    2 seconds per epoch on a K520 GPU.
    '''

    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import RMSprop

    batch_size = 128
    num_classes = 10
    epochs = 20

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    callbacks = [Visualizer(x=x_test)]

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == "__main__":
    test()