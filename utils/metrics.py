import numpy as np
import tensorflow as tf


def convert_to_unit_vector(angles):
    x = -tf.math.cos(angles[:, 0]) * tf.math.sin(angles[:, 1])
    y = -tf.math.sin(angles[:, 0])
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