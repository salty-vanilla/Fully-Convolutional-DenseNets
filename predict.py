from data_generator import DataGenerator
from keras.models import model_from_json
from densenet_fc import DenseNetFCN
from keras import backend as K
import argparse
import numpy as np


width = 224
height = 224
channel = 3
nb_classes = 21


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_list', type=str)
    parser.add_argument('image_dir', type=str)
    parser.add_argument('label_dir', type=str)
    parser.add_argument('param_path', type=str)

    args = parser.parse_args()

    test_list = args.test_list
    test_image_dir = args.image_dir
    test_label_dir = args.label_dir
    model_weights = args.param_path

    input_shape = (height, width, channel) if K.image_dim_ordering() == 'tf' \
        else (channel, height, width)
    model = DenseNetFCN(input_shape, nb_dense_block=5, growth_rate=16,
                        nb_layers_per_block=4, upsampling_type='upsampling',
                        classes=nb_classes, activation='softmax')

    # jsonからロードできない ・・・
    model.load_weights(model_weights)

    test_names = [name.rstrip('\r\n') for name in open(test_list).readlines()][:50]
    test_generator = DataGenerator(file_names=test_names, image_dir=test_image_dir, label_dir=test_label_dir,
                                   size=(width, height), nb_classes=nb_classes)

    images, labels = test_generator.next_batch(len(test_names)).next()
    predicted = model.predict(images, batch_size=10)

    np.save('predicted.npy', predicted)
    np.save('image.npy', images)
    np.save('label.py', labels)


if __name__ == "__main__":
    main()
