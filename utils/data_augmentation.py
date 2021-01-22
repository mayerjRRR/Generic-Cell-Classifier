import tensorflow as tf
# from tensorflow_addons.image import rotate


def augment_image(image, crop_size):
    image = crop(image, crop_size)
    image = flip(image)
    image = _rotate(image)
    image = light(image)
    return image


def light(image):
    image = tf.image.random_brightness(image, 0.2)
    return image


def _rotate(image):
    # random_angle = tf.random.uniform([], -180, 180)
    # image = rotate(image, random_angle, interpolation="BILINEAR")
    return image


def flip(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image


def crop(image, crop_size):
    image = tf.image.random_crop(image, [crop_size, crop_size, 3])
    return image
