from datetime import datetime

import tensorflow as tf
from tensorflow import GradientTape
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.metrics import sparse_categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tqdm import tqdm

from classifier import wrap_in_softmax, build_mobilenet_classifier
from data_loader import DataLoader

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)

model = build_mobilenet_classifier()
optimizer = Adam()
batch_size = 32
epochs = 15
step = 0

training_dataset = DataLoader(batch_size, "data").train_dataset

loss_metric = tf.keras.metrics.Mean()
accuracy_metric = tf.keras.metrics.Mean()

train_iterator = tqdm(training_dataset)
for samples, labels in train_iterator:
    with GradientTape() as tape:
        predictions = model(samples)
        loss = sparse_categorical_crossentropy(labels, predictions, from_logits=True)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    accuracy = sparse_categorical_accuracy(labels, predictions)

    loss_metric.update_state(tf.reduce_mean(loss))
    accuracy_metric.update_state(tf.reduce_mean(accuracy))

    if step % 50 == 0:
        with file_writer.as_default():
            tf.summary.scalar('Loss', loss_metric.result(), step=step)
            tf.summary.scalar('Accuracy', accuracy_metric.result(), step=step)
            loss_metric.reset_states()
            accuracy_metric.reset_states()
    if step % 500 == 0:
        softmax_model = wrap_in_softmax(model)
        softmax_model.save(logdir + "/model.h5")

    step += 1

    train_iterator.set_postfix_str(
        f'batch_loss={float(loss.numpy().mean()):.4f}, '
        f'batch_acc={float(accuracy.numpy().mean()):.4f}'
    )
    train_iterator.refresh()
