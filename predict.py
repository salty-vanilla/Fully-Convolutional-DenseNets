from data_generator import DataGenerator
from keras.models import model_from_json


def main():
    model_json = "dense_fcn.json"
    model_weights = "dense_fcn.hdf5"
    model = model_from_json(model_json)
    model.load_weights(model_weights)


if __name__ == "__main__":
    main()
