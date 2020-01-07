import os
import pickle
import re
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from settings import DATA_PATH

plt.ioff()


def load_one_person_one_feature(person, feature):
    dataset_path = os.path.join(DATA_PATH, "mpiigaze_processed_both_rgb")
    feature_path = os.path.join(dataset_path, feature)
    filenames = os.listdir(feature_path)
    filenames[:] = [filename for filename in filenames if re.search(person, filename) is not None]
    data = list()
    for filename in filenames:
        file_path = os.path.join(feature_path, filename)
        with open(file_path, "rb") as file:
            file_data = pickle.load(file)
            data.append(file_data)
    data = np.concatenate(data)
    return data


def _load_one_person(person="p00", grayscale=False):
    dirs = ["right_eye", "left_eye", "headpose", "gaze", "coordinates"]
    data = dict()
    for d in dirs:
        data[d] = load_one_person_one_feature(person=person, feature=d)

    data["right_eye"] = _prepare_images(data["right_eye"], grayscale)
    data["left_eye"] = _prepare_images(data["left_eye"], grayscale)
    data["headpose"] = _prepare_headposes(data["headpose"])

    return data


def rgb2gray(rgb):
    r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).reshape((rgb.shape[0], rgb.shape[1], rgb.shape[2], 1))
    return gray


def _prepare_images(images, grayscale):
    if grayscale:
        images = rgb2gray(images)
    images = images / 255.
    return images


def _prepare_headposes(headposes):
    headposes[:, 3:] = headposes[:, 3:] / 100.
    return headposes


def _prepare_dataset_generator(x, y, batch_size=128):
    """Prepares dataset to train in mini-batches."""
    ds_size = len(y)
    x = tf.data.Dataset.from_tensor_slices(x)
    y = tf.data.Dataset.from_tensor_slices(y)
    train_dataset = tf.data.Dataset.zip((x, y))
    buffer_size = ds_size if ds_size <= 32768 else 32768
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size=batch_size)
    return train_dataset


def load_mpiigaze_train_test_ds_generator(person, out_class, val_split = 0.2, batch_size = 128, grayscale=False):
    dataset = _load_one_person(person, grayscale)
    y = dataset[out_class][:, 0].squeeze()
    (right_eye_train, right_eye_test,
     left_eye_train, left_eye_test,
     headpose_train, headpose_test,
     y_train, y_test) = train_test_split(dataset["right_eye"], dataset["left_eye"],
                                                          dataset["headpose"], y,
                                                          test_size=val_split, random_state=42)
    del dataset
    train_dataset = _prepare_dataset_generator((right_eye_train, left_eye_train, headpose_train), y_train, batch_size)
    test_dataset = _prepare_dataset_generator((right_eye_test, left_eye_test, headpose_test), y_test, batch_size)
    return train_dataset, test_dataset