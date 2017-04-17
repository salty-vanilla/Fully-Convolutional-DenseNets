from keras.models import model_from_json
from keras import backend as K
import argparse
import numpy as np
import os
try:
    import pickle
except:
    import cPickle as pickle
from PIL import Image

from result_utils import visualize_result
from data_generator import DataGenerator
from densenet_fc import DenseNetFCN


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
    parser.add_argument('--batch_size', '-bs', type=int, default=10)
    parser.add_argument('--result_dir', '-rd', type=str, default='./result')
    parser.add_argument('--palette', '-p', type=str, default='palette.pkl')

    args = parser.parse_args()

    test_list = args.test_list
    test_image_dir = args.image_dir
    test_label_dir = args.label_dir
    model_weights = args.param_path
    batch_size = args.batch_size
    result_dir = args.result_dir
    palette = args.palette

    os.makedirs(result_dir, exist_ok=True)

    input_shape = (height, width, channel) if K.image_dim_ordering() == 'tf' \
        else (channel, height, width)
    model = DenseNetFCN(input_shape, nb_dense_block=5, growth_rate=16,
                        nb_layers_per_block=4, upsampling_type='upsampling',
                        classes=nb_classes, activation='softmax')

    # jsonからロードできない ・・・
    model.load_weights(model_weights)

    test_names = [name.rstrip('\r\n') for name in open(test_list).readlines()][:30]
    test_generator = DataGenerator(file_names=test_names, image_dir=test_image_dir, label_dir=test_label_dir,
                                   size=(width, height), nb_classes=nb_classes)

    for iter_, batch in enumerate(test_generator.next_batch(batch_size)):
        datas, labels = batch
        predicted = model.predict_on_batch(datas)

        images = (datas * 255).astype('uint8')
        # results = visualize_result(predicted, palette)

        for idx in range(len(images)):
            image = Image.fromarray(images[idx])
            result = visualize_result(predicted[idx], palette)
            image.save(os.path.join(result_dir, 'input_{}.png'.format(idx + len(images) * iter_)))
            result.save(os.path.join(result_dir, 'result_{}.png'.format(idx + len(images) * iter_)))

        if idx + idx * iter_ + 1 >= len(test_names):
            break


if __name__ == "__main__":
    main()
