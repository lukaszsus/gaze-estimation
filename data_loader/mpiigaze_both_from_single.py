# coding: utf-8

import os
import numpy as np
from sklearn.model_selection import train_test_split

from data_loader.mpiigaze_processed_loader import prepare_dataset, _prepare_images, _prepare_headposes, _prepare_gaze
from user_settings import DATA_PATH
from utils.configs import USE_FLOAT64

NUMBER_OF_SUBJECTS = 15


def _load_one_person(dataset_name, person, grayscale):
    """
    :person: str or int - id of person
    """
    path = f"{dataset_name}/p{str(person).zfill(2)}.npz"
    path = os.path.join(os.path.join(DATA_PATH, path))
    with np.load(path) as fin:
        right_images = _prepare_images(fin['right_image'], grayscale)
        left_images = _prepare_images(fin['left_image'], grayscale)
        poses = _prepare_headposes(fin['pose'], normalize=False)
        gazes = _prepare_gaze(fin['gaze'])

    if USE_FLOAT64:
        return right_images.astype(np.float64), left_images.astype(np.float64), poses.astype(np.float64), gazes.astype(np.float64)
    return right_images, left_images, poses, gazes


def _load_all_people(dataset_name, grayscale):
    right_images_all = list()
    left_images_all = list()
    poses_all = list()
    gazes_all = list()
    subject_ids = list()
    for i in range(NUMBER_OF_SUBJECTS):
        right_images, left_images, poses, gazes = _load_one_person(dataset_name, i, grayscale)
        right_images_all.append(right_images)
        left_images_all.append(left_images)
        poses_all.append(poses)
        gazes_all.append(gazes)
        subject_ids.append(np.array([i] * len(poses)))
    right_images_all = np.concatenate(right_images_all, axis=0)
    left_images_all = np.concatenate(left_images_all, axis=0)
    poses_all = np.concatenate(poses_all, axis=0)
    gazes_all = np.concatenate(gazes_all, axis=0)
    subject_ids = np.concatenate(subject_ids, axis=0)

    return right_images_all, left_images_all, poses_all, gazes_all, subject_ids


def load_mpiigaze_train_test_ds_both_from_single(dataset_name, person_id, val_split=0.2, batch_size=128,
                                                 grayscale=True):
    test_subject_ids = None
    if person_id is None:
        right_images, left_images, poses, gazes, subject_ids = _load_all_people(dataset_name, grayscale)
        (right_eye_train, right_eye_test,
         left_eye_train, left_eye_test,
         headpose_train, headpose_test,
         y_train, y_test,
         train_subject_ids, test_subject_ids) = train_test_split(right_images, left_images, poses, gazes, subject_ids,
                                                                 test_size=val_split,
                                                                 random_state=42, stratify=subject_ids)
    else:
        right_images, left_images, poses, gazes = _load_one_person(dataset_name, person_id, grayscale)
        (right_eye_train, right_eye_test,
         left_eye_train, left_eye_test,
         headpose_train, headpose_test,
         y_train, y_test) = train_test_split(right_images, left_images, poses, gazes, test_size=val_split,
                                             random_state=42)

    train_dataset = prepare_dataset((right_eye_train, left_eye_train, headpose_train), y_train, batch_size)
    # don't shuffle test dataset to have consistent outcomes
    test_dataset = prepare_dataset((right_eye_test, left_eye_test, headpose_test), y_test, batch_size, shuffle=False)

    return train_dataset, test_dataset, test_subject_ids


