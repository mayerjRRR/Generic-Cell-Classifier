import tensorflow as tf

from utils.csv_tools import get_file_name_and_label_from_csv
from utils.data_augmentation import augment_image


def load_image(crop_size):
    def _load_image(file_name, label):
        image = load_actual_image(file_name, crop_size)
        return image, label

    return _load_image


def load_actual_image(file_name, crop_size):
    image = tf.io.read_file(file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1

    image = augment_image(image, crop_size)

    return image


class DataLoader:
    def __init__(self, batchsize, dataset_directory, supported_microscopes, crop_size):
        self.buffer_size = 100
        self.batch_size = batchsize
        self.dataset_directory = dataset_directory
        self.train_dataset = self.generate_training_dataset(supported_microscopes, crop_size)

    def generate_training_dataset(self, supported_microscopes, crop_size):
        image_paths_and_labels = get_file_name_and_label_from_csv(self.dataset_directory, "data.csv",
                                                                  supported_microscopes)
        train_dataset = tf.data.Dataset.from_tensor_slices(image_paths_and_labels)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.shuffle(self.buffer_size)
        train_dataset = train_dataset.map(load_image(crop_size),
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(self.batch_size)
        return train_dataset
