import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import CSVLogger
import argparse
import os

from data_generator import DataGenerator
from densenet_fc import DenseNetFCN
from callbacks import BatchLogger, ModelSaver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_list', type=str)
    parser.add_argument('image_dir', type=str)
    parser.add_argument('label_dir', type=str)
    parser.add_argument('--valid_list', '-vl', type=str, default=None)
    parser.add_argument('--valid_image_dir', '-vid', type=str, default=None)
    parser.add_argument('--valid_label_dir', '-vld', type=str, default=None)
    parser.add_argument('--width', '-w', type=int, default=224)
    parser.add_argument('--height', '-h', type=int, default=224)
    parser.add_argument('--channel', '-ch', type=int, default=3)
    parser.add_argument('--batch_size', '-b', type=int, default=10)
    parser.add_argument('--nb_epoch', '-e', type=int, default=30)
    parser.add_argument('--nb_classes', 'cl', type=int, default=21)
    parser.add_argument('--param_dir', '-pd', type=str, default="./params")

    args = parser.parse_args()
    train_list = args.train_list
    train_image_dir = args.image_dir
    train_label_dir = args.label_dir

    valid_list = args.valid_list
    valid_image_dir = args.valid_image_dir
    valid_label_dir = args.valid_label_dir

    width = args.width
    height = args.height
    channel = args.channel
    batch_size = args.batch_size
    nb_epoch = args.nb_epoch
    nb_classes = args.nb_classes
    param_dir = args.param_dir

    callbacks = [CSVLogger("learning_log_epoch.csv"),
                 BatchLogger("learning_log_iter.csv"),
                 ModelSaver(os.path.join(param_dir, "dense_fcn_{epoch:02d}.hdf5"),
                            save_freq=5)]

    input_shape = (height, width, channel) if K.image_dim_ordering() == 'tf' \
        else (channel, height, width)

    dense_fcn = DenseNetFCN(input_shape)

    opt = Adam(lr=1e-5, beta_1=0.1)
    dense_fcn.compile(opt, 'categorical_cross_entropy', metrics=['accuracy'])

    train_names = [name for name in open(train_list).readlines()]
    train_gen = DataGenerator(file_names=train_names, image_dir=train_image_dir, label_dir=train_label_dir,
                              size=(width, height), nb_classes=nb_classes)

    if valid_list is not None:
        valid_names = [name for name in open(valid_list).readlines()]
        valid_gen = DataGenerator(file_names=valid_names, image_dir=valid_image_dir, label_dir=valid_label_dir,
                                  size=(width, height), nb_classes=nb_classes)
        dense_fcn.fit_generator(train_gen.next_batch(batch_size),
                                samples_per_epoch=train_gen.data_num,
                                epochs=nb_epoch,
                                validation_data=valid_gen.next_batch(batch_size),
                                callbacks=callbacks)
    else:
        dense_fcn.fit_generator(train_gen.next_batch(batch_size),
                                samples_per_epoch=train_gen.data_num,
                                epochs=nb_epoch,
                                callbacks=callbacks)

    dense_fcn.save_weights("dense_fcn.hdf5")


if __name__ == "__main__":
    main()
