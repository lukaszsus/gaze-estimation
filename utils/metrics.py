import numpy as np
import tensorflow as tf
import time


def convert_to_unit_vector(angles):
    x = -tf.math.cos(angles[:, 0]) * tf.math.sin(angles[:, 1])
    y = -tf.math.sin(angles[:, 0])
    # z = -tf.math.cos(angles[:, 0]) * tf.math.cos(angles[:, 1])
    z = -tf.math.cos(angles[:, 1]) * tf.math.cos(angles[:, 1])
    norm = tf.math.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


def compute_angle_error(labels, predictions):
    """
    Computes mean angle error between two vectors.
    Uses tensorflow.
    """
    pred_x, pred_y, pred_z = convert_to_unit_vector(predictions)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return tf.math.acos(angles) * 180 / np.pi


def final_predictions(model, test_dataset):
    predictions = list()
    labels = list()
    for x, y in test_dataset:
        predictions.append(model.predict(x))
        labels.append(y)
    predictions = tf.concat(predictions, axis=0)
    labels = tf.concat(labels, axis=0)
    return labels, predictions


def final_test_measure_time(model, test_dataset):
    """
    Measures quality of model on test dataset after all epochs.
    """
    # for measuring forward pass time it is required to have batch size equals to 1
    test_dataset_batch_size_1 = tf.data.Dataset.zip(test_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))).take(100).batch(batch_size=1)
    predictions = list()
    forward_pass_time = time.time()
    for x, y in test_dataset_batch_size_1:
        predictions.append(model.predict(x))
    forward_pass_time = time.time() - forward_pass_time
    forward_pass_time = forward_pass_time / len(predictions)
    del test_dataset_batch_size_1
    return forward_pass_time
