import os
import PIL
from PIL import Image
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator

image_ext = 'jpg'
label_ext = 'png'


class DataIterator(Iterator):
    def __init__(self, file_names, image_dir, label_dir, size, nb_classes, batch_size, shuffle, seed=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.names = file_names
        self.size = size
        self.nb_classes = nb_classes
        self.data_num = len(self.names)
        self.image_paths = np.array([os.path.join(self.image_dir, name + '.' + image_ext) for name in self.names])
        self.label_paths = np.array([os.path.join(self.label_dir, name + '.' + label_ext) for name in self.names])
        super().__init__(self.data_num, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        image_path_batch = self.image_paths[index_array]
        label_path_batch = self.label_paths[index_array]
        image_batch = np.array([self.load_image(path, mode="data") for path in image_path_batch])
        label_batch = np.array([self.load_image(path, mode="label") for path in label_path_batch])
        return image_batch, label_batch

    def load_image(self, path, mode="original", is_color=True, crop_thr=0.7):
        assert mode in ["original", "data", "label"]

        if mode == "label":
            resample = PIL.Image.NEAREST
        else:
            resample = PIL.Image.BILINEAR

        image = Image.open(path)

        if not is_color:
            image = image.convert('L')

        # 指定されたサイズと異なるならリサイズ
        if self.size != image.size:
            w, h = image.size
            # width と height の比率が近い場合はリサイズ
            # 遠い場合はクロップする
            if min([w, h]) / max([w, h]) > crop_thr:
                image = image.resize(self.size, resample)
            else:
                image = crop_image(image, self.size, resample=resample)

        if mode == "original":
            return image

        elif mode == "data":
            image = np.array(image, dtype='float32')
            if not is_color:
                image = image.reshape((image.shape[0], image.shape[1], 1))

            image = preprocess(image)

            if K.image_dim_ordering() == 'th':
                image = image.transpose((2, 0, 1))

            return image

        else:
            labels = np.array(image, dtype='int32')

            # 255の画素はBGクラスにしてしまう
            mask = (labels == 255)
            labels[mask] = 0

            one_hot = one_hot_vectorize(labels, self.nb_classes)

            if K.image_dim_ordering() == 'th':
                one_hot = one_hot.transpose((2, 0, 1))

            return one_hot


class DataGenerator:
    def __init__(self, file_names, image_dir, label_dir, size, nb_classes):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.names = file_names
        self.size = size
        self.nb_classes = nb_classes
        self.data_num = len(self.names)

    def next_batch(self, batch_size, is_shuffle=True):
        return DataIterator(file_names=self.names,
                            image_dir=self.image_dir, label_dir=self.label_dir,
                            size=self.size, nb_classes=self.nb_classes,
                            batch_size=batch_size, shuffle=is_shuffle)


def crop_image(image, size, resample=PIL.Image.BILINEAR):
    target_width, target_height = size
    width, height = image.size
    _image = image.copy()

    if width < height:
        if width < target_width:
            _image = image.resize((target_width, target_height * height // width), resample)
            width, height = _image.size
    else:
        if height < target_height:
            _image = image.resize((target_width * width // height, target_height), resample)
            width, height = _image.size

    _image = _image.crop((int((width - target_width) * 0.5), int((height - target_height) * 0.5),
                          int((width + target_width) * 0.5), int((height + target_height) * 0.5)))

    return _image


def preprocess(image):
    return image.astype('float32') / 255


def one_hot_vectorize(labels, nb_classes):
    h, w = labels.shape
    one_hot = np.zeros((h, w, nb_classes))

    for j in range(h):
        for i in range(w):
            one_hot[j, i, labels[j, i]] = 1
    return one_hot
