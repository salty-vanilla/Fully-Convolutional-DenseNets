# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import keras.callbacks
try:
    import cPickle as pickle
except:
    import pickle

class HistorySaver(keras.callbacks.Callback):
    '''Callback that records events
    into a `History` object.
    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    '''

    def __init__(self, dst_path):
        self.path = dst_path

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        loss = self.history['loss']
        val_loss = self.history['val_loss']
        acc = self.history["acc"]
        val_acc = self.history["val_acc"]

        result = (loss, val_loss, acc, val_acc)
        save_file = open(self.path, "wb")
        pickle.dump(result, save_file, -1)
        save_file.close()


class ModelCheckpointEx(keras.callbacks.ModelCheckpoint):
    def __init__(self, dst_path, verbose=0, save_freq=1):
        super(ModelCheckpointEx, self).__init__(dst_path, verbose=verbose)
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.save_freq == 0:
            super(ModelCheckpointEx, self).on_epoch_end(epoch, logs=logs)