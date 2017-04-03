import numpy as np
from data_generator import one_hot_vectorize
from PIL import Image, ImagePalette
try:
    import pickle
except:
    import cPickle as pickle


def convert_to_label(predicted):
    height, width = predicted.shape[:2]
    label = np.zeros(shape=(height, width))
    for j in range(height):
        for i in range(width):
            label[j, i] = np.argmax(predicted[j, i])
    return label


def extract_palette(dst_path, src_image_path):
    image = Image.open(src_image_path)
    with open(dst_path, 'wb') as f:
        pickle.dump(image.palette, f)


def get_palette(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def visualize_result(predicted, palette):
    label = convert_to_label(predicted)
    pal = get_palette(palette)

    pil_label = Image.fromarray(np.uint8(label), mode='P')
    pil_label.palette = pal
    pil_label.show()
    return pil_label


if __name__ == "__main__":
    extract_palette("palette.pkl", "2007_000032.png")
    label = np.zeros(shape=(224, 224), dtype='uint8')
    label[0:100, 0:100] = 5
    predicted = one_hot_vectorize(label, 21)
    result = visualize_result(predicted, "palette.pkl")

