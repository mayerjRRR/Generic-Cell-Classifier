import glob
from os.path import join
import numpy as np
import tensorflow as tf

from classifier import INPUT_DIMENSION


def load_image(file_name, label):
    image = load_actual_image(file_name)
    return image, label


def load_actual_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    #TODO: random crop
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
        image_paths = glob.glob(join(self.dataset_directory, '*.jpg'))
        image_labels = [get_label_from_filename(file_name) for file_name in image_paths]
        train_dataset = tf.data.Dataset.from_tensor_slices((image_paths,image_labels))
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.map(load_image,
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.buffer_size)
        train_dataset = train_dataset.batch(self.batch_size)
        return train_dataset

    def get_file_name_and_label_from_csv(self, csv_path):
        #TODO: actually fill with meaning
        return [],[]

