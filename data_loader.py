import os
import tensorflow as tf
from classifier import INPUT_DIMENSION

from csv_tools import get_file_name_and_label_from_csv


def load_image(file_name, label):
    image = load_actual_image(file_name)
    return image, label


def load_actual_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    #TODO: random crop: https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/RandomCrop
    image = tf.image.resize(image, [INPUT_DIMENSION, INPUT_DIMENSION])
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


class DataLoader:
    def __init__(self, batchsize, dataset_directory):
        self.buffer_size = 100
        self.batch_size = batchsize
        self.dataset_directory = dataset_directory
        self.train_dataset = self.generate_training_dataset()

    def generate_training_dataset(self):
        image_paths_and_labels = self.get_file_name_and_label_from_csv(os.join.path("data", "piss.csv"))
        train_dataset = tf.data.Dataset.from_tensor_slices(image_paths_and_labels)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.map(load_image,
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.buffer_size)
        train_dataset = train_dataset.batch(self.batch_size)
        return train_dataset

