# coding: utf-8

import os
import numpy as np
from sklearn.model_selection import train_test_split

from data_loader.mpiigaze_processed_loader import prepare_dataset
from user_settings import DATA_PATH
from utils.configs import USE_FLOAT64

NUMBER_OF_SUBJECTS = 15


def _load_one_person(person):
    """
    :person: str or int - id of person
    """
    path = f"hysts_mpiigazed_processed/p{str(person).zfill(2)}.npz"
    path = os.path.join(os.path.join(DATA_PATH, path))
    with np.load(path) as fin:
        images = fin['image']
        shape = images.shape
        images = images.reshape((shape[0], shape[1], shape[2], 1))
        poses = fin['pose']
        gazes = fin['gaze']

    if USE_FLOAT64:
        return images.astype(np.float64), poses.astype(np.float64), gazes.astype(np.float64)
    return images, poses, gazes


def _load_all_people():
    images_all = list()
    poses_all = list()
    gazes_all = list()
    subject_ids = list()
    for i in range(NUMBER_OF_SUBJECTS):
        images, poses, gazes = _load_one_person(i)
        images_all.append(images)
        poses_all.append(poses)
        gazes_all.append(gazes)
        subject_ids.append(np.array([i] * len(poses)))
    images_all = np.concatenate(images_all, axis=0)
    poses_all = np.concatenate(poses_all, axis=0)
    gazes_all = np.concatenate(gazes_all, axis=0)
    subject_ids = np.concatenate(subject_ids, axis=0)

    return images_all, poses_all, gazes_all, subject_ids


def load_hysts_mpiigaze_train_test_ds(person_id, val_split=0.2, batch_size=128):
    test_subject_ids = None
    if person_id is None:
        images, poses, gazes, subject_ids = _load_all_people()
        (eye_train, eye_test,
         headpose_train, headpose_test,
         y_train, y_test,
         train_subject_ids, test_subject_ids) = train_test_split(images, poses, gazes, subject_ids, test_size=val_split,
                                                                 random_state=42, stratify=subject_ids)
    else:
        images, poses, gazes = _load_one_person(person_id)
        (eye_train, eye_test,
         headpose_train, headpose_test,
         y_train, y_test) = train_test_split(images, poses, gazes, test_size=val_split, random_state=42)

    train_dataset = prepare_dataset((eye_train, headpose_train), y_train, batch_size)
    # don't shuffle test dataset to have consistent outcomes
    test_dataset = prepare_dataset((eye_test, headpose_test), y_test, batch_size, shuffle=False)

    return train_dataset, test_dataset, test_subject_ids


