import numpy as np
from data_generator import one_hot_vectorize


def argmax_image(predicted):
    height, width = predicted.shape[:2]
    label = np.zeros(shape=(height, width))
    for j in range(height):
        for i in range(width):
            label[j, i] = np.argmax(predicted[j, i])
    return label


if __name__ == "__main__":
    label = np.random.randint(0, 21, size=(224, 224))
    predicted = one_hot_vectorize(label, 21)
    print(argmax_image(predicted))
