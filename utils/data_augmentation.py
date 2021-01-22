import tensorflow as tf
import tensorflow_addons

from trainer import INPUT_DIMENSION


def augment_image(image):
    image = crop(image)
    image = flip(image)
    image = _rotate(image)
    image = light(image)
    return image


def light(image):
    image = tf.image.random_brightness(image, 0.2)
    return image


def _rotate(image):
    random_angle = tf.random.uniform([], -180, 1800)
    image = tensorflow_addons.image.rotate(image, random_angle, interpolation="BILINEAR")
    return image


def flip(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image


def crop(image):
    image = tf.image.random_crop(image, [INPUT_DIMENSION, INPUT_DIMENSION, 3])
    return image
