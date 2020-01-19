import os
import pickle
import re
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from settings import DATA_PATH
from utils.configs import USE_FLOAT64

plt.ioff()
NUMBER_OF_SUBJECTS = 15
dirs = ["right_eye", "left_eye", "headpose", "gaze", "coordinates"]


def load_one_person_one_feature(dataset_name, person_id, feature):
    dataset_path = os.path.join(DATA_PATH, dataset_name)
    feature_path = os.path.join(dataset_path, feature)
    filenames = os.listdir(feature_path)
    person_id_str = "p" + str(person_id).zfill(2)
    filenames[:] = [filename for filename in filenames if re.search(person_id_str, filename) is not None]
    data = list()
    for filename in filenames:
        file_path = os.path.join(feature_path, filename)
        with open(file_path, "rb") as file:
            file_data = pickle.load(file)
            data.append(file_data)
    data = np.concatenate(data)
    return data


def _load_one_person(dataset_name, person_id=0, grayscale=False, normalize_headpose=False):
    data = dict()
    for d in dirs:
        data[d] = load_one_person_one_feature(dataset_name=dataset_name, person_id=person_id, feature=d)

    data["right_eye"] = _prepare_images(data["right_eye"], grayscale)
    data["left_eye"] = _prepare_images(data["left_eye"], grayscale)
    data["headpose"] = _prepare_headposes(data["headpose"], normalize=normalize_headpose)
    data["gaze"] = _prepare_gaze(data["gaze"])
    data["coordinates"] = _prepare_coordinates(data["coordinates"])


    return data


def _load_all_people(dataset_name, grayscale=False, normalize_headpose=False):
    data = dict()
    for d in dirs:
        data[d] = list()

    subject_ids = list()
    for i in range(NUMBER_OF_SUBJECTS):
        one_person_data = _load_one_person(dataset_name, i, grayscale, normalize_headpose)
        for d in dirs:
            data[d].append(one_person_data[d])
        subject_ids.append(np.array([i] * len(data["gaze"])))

    for d in dirs:
        data[d] = np.concatenate(data[d], axis=0)

    return data, subject_ids


def rgb2gray(rgb):
    r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).reshape((rgb.shape[0], rgb.shape[1], rgb.shape[2], 1))
    return gray


def _prepare_images(images, grayscale):
    if grayscale:
        images = rgb2gray(images)
    images = images / 255.
    if USE_FLOAT64:
        return images.astype(np.float64)
    else:
        return images.astype(np.float32)


def _prepare_headposes(headposes, normalize):
    # headposes[:, 3:] = headposes[:, 3:]
    if normalize:
        headposes = headposes / 100.
    if USE_FLOAT64:
        return headposes.astype(np.float64)
    else:
        return headposes.astype(np.float32)


def _prepare_gaze(gaze):
    if USE_FLOAT64:
        return gaze.astype(np.float64)
    else:
        return gaze.astype(np.float32)


def _prepare_coordinates(coordinates):
    if USE_FLOAT64:
        return coordinates.astype(np.float64)
    else:
        return coordinates.astype(np.float32)


def prepare_dataset(x, y, batch_size=128, shuffle=True):
    """Prepares dataset to train in mini-batches."""
    ds_size = len(y)
    x = tf.data.Dataset.from_tensor_slices(x)
    y = tf.data.Dataset.from_tensor_slices(y)
    dataset = tf.data.Dataset.zip((x, y))
    if shuffle:
        buffer_size = ds_size if ds_size <= 32768 else 32768
        dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size=batch_size)
    else:
        dataset = dataset.batch(batch_size=batch_size)
    return dataset


def load_mpiigaze_train_test_ds(dataset_name, person_id, out_class, val_split=0.2, batch_size=128, grayscale=False,
                                normalize_headpose=True):
    """
    :out_class: 'gaze' or 'coordinates'
    :normalize_headpose: divide by 100 - use when headpose is a vector not angle
    """
    test_subject_ids = None
    if person_id is None:
        dataset, subject_ids = _load_all_people(dataset_name, grayscale, normalize_headpose)
        y = dataset[out_class]
        (right_eye_train, right_eye_test,
         left_eye_train, left_eye_test,
         headpose_train, headpose_test,
         y_train, y_test,
         train_subject_ids, test_subject_ids) = train_test_split(dataset["right_eye"], dataset["left_eye"],
                                             dataset["headpose"], y, test_subject_ids,
                                             test_size=val_split, random_state=42,
                                             stratify=subject_ids)
    else:
        dataset = _load_one_person(dataset_name, person_id, grayscale, normalize_headpose)
        y = dataset[out_class]
        (right_eye_train, right_eye_test,
         left_eye_train, left_eye_test,
         headpose_train, headpose_test,
         y_train, y_test) = train_test_split(dataset["right_eye"], dataset["left_eye"],
                                             dataset["headpose"], y,
                                             test_size=val_split, random_state=42)
    del dataset
    train_dataset = prepare_dataset((right_eye_train, left_eye_train, headpose_train), y_train, batch_size)
    test_dataset = prepare_dataset((right_eye_test, left_eye_test, headpose_test), y_test, batch_size)
    return train_dataset, test_dataset, test_subject_ids


def load_mpiigaze_train_test_ds_2_vectors_headpose(dataset_name, person_id, out_class, val_split=0.2, batch_size=128, grayscale=False):
    return load_mpiigaze_train_test_ds(dataset_name, person_id, out_class, val_split=val_split, batch_size=batch_size,
                                       grayscale=grayscale, normalize_headpose=True)


def load_mpiigaze_train_test_2_angles_headpose(dataset_name, person_id, out_class, val_split=0.2, batch_size=128, grayscale=False):
    return load_mpiigaze_train_test_ds(dataset_name, person_id, out_class, val_split=val_split, batch_size=batch_size,
                                       grayscale=grayscale, normalize_headpose=False)
