from data_generator import DataGenerator
from keras.models import model_from_json
from densenet_fc import DenseNetFCN
from keras import backend as K


width = 224
height = 224
channel = 3
nb_classes = 21


def main():
    input_shape = (height, width, channel) if K.image_dim_ordering() == 'tf' \
        else (channel, height, width)
    model = DenseNetFCN(input_shape, nb_dense_block=5, growth_rate=16,
                        nb_layers_per_block=4, upsampling_type='upsampling',
                        classes=nb_classes, activation='softmax')

    # jsonからロードできない ・・・
    model_weights = "dense_fcn.hdf5"
    model.load_weights(model_weights)

if __name__ == "__main__":
    main()
