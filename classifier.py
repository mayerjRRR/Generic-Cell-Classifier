import tensorflow as tf
from tensorflow.keras import layers



def build_mobilenet_classifier(number_of_classes, crop_size):
    mobilenet_stump = tf.keras.applications.MobileNetV2(input_shape=(crop_size, crop_size, 3),
                                                        include_top=False,
                                                        weights='imagenet')
    input = tf.keras.Input((crop_size, crop_size, 3))
    mobilenet_features = mobilenet_stump(input)
    x = tf.keras.layers.Flatten()(mobilenet_features)
    mobilenet_stump.trainable = False
    x = layers.Dense(512, use_bias=True)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dense(number_of_classes, use_bias=True)(x)

    return tf.keras.Model(inputs=input, outputs=x)


def wrap_in_softmax(model: tf.keras.Model):
    input = tf.keras.Input(shape=model.input_shape[1:])
    x = model(input)
    x = tf.keras.layers.Softmax()(x)
    return tf.keras.Model(inputs=input, outputs=x)
