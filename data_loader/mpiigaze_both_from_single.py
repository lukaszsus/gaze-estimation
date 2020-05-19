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
        return right_images.astype(np.float64), left_images.astype(np.float64), poses.astype(np.float64), gazes.astype(
            np.float64)
    return right_images, left_images, poses, gazes


def _load_all_people(dataset_name, grayscale, leave_person_id=None):
    right_images_all = list()
    left_images_all = list()
    poses_all = list()
    gazes_all = list()
    subject_ids = list()
    for i in range(NUMBER_OF_SUBJECTS):
        if leave_person_id is not None and leave_person_id == i:
            continue
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


############## LEAVE ONE OUT

def load_mpiigaze_train_test_ds_both_leave_one_out(dataset_name, person_id, val_split=0.2, batch_size=128,
                                                   grayscale=True):
    """
    val_split argument only for consistent function interface
    """
    right_eye_train, left_eye_train, headpose_train, y_train, train_subject_ids = _load_all_people(dataset_name,
                                                                                                   grayscale, person_id)
    right_eye_test, left_eye_test, headpose_test, y_test = _load_one_person(dataset_name, person_id, grayscale)

    test_subject_ids = np.asarray([person_id] * len(y_test))

    train_dataset = prepare_dataset((right_eye_train, left_eye_train, headpose_train), y_train, batch_size)
    # don't shuffle test dataset to have consistent outcomes
    test_dataset = prepare_dataset((right_eye_test, left_eye_test, headpose_test), y_test, batch_size, shuffle=False)

    return train_dataset, test_dataset, test_subject_ids


############## REJECT SUSPICIOUS

def load_mpiigaze_train_test_ds_both_leave_one_out_reject_suspicious(dataset_name, person_id, val_split=0.2,
                                                                     batch_size=128,
                                                                     grayscale=True):
    """
    val_split argument only for consistent function interface
    """
    right_eye_train, left_eye_train, headpose_train, y_train, train_subject_ids = _load_all_people_reject_suspicious(
        dataset_name, grayscale, person_id)
    right_eye_test, left_eye_test, headpose_test, y_test = _load_one_person(dataset_name, person_id, grayscale)

    if person_id == 2:
        right_eye_test, left_eye_test, headpose_test, y_test = _filter_subject_02(
            right_eye_test, left_eye_test, headpose_test, y_test)
    if person_id == 10:
        right_eye_test, left_eye_test, headpose_test, y_test = _filter_subject_10(
            right_eye_test, left_eye_test, headpose_test, y_test)

    test_subject_ids = np.asarray([person_id] * len(y_test))

    train_dataset = prepare_dataset((right_eye_train, left_eye_train, headpose_train), y_train, batch_size)
    # don't shuffle test dataset to have consistent outcomes
    test_dataset = prepare_dataset((right_eye_test, left_eye_test, headpose_test), y_test, batch_size, shuffle=False)

    return train_dataset, test_dataset, test_subject_ids


def _load_all_people_reject_suspicious(dataset_name, grayscale, leave_person_id=None):
    right_images_all = list()
    left_images_all = list()
    poses_all = list()
    gazes_all = list()
    subject_ids = list()
    for i in range(NUMBER_OF_SUBJECTS):
        if leave_person_id is not None and leave_person_id == i:
            continue

        # filter suspicious
        if i == 7:
            continue
        right_images, left_images, poses, gazes = _load_one_person(dataset_name, i, grayscale)
        if i == 2:
            right_images, left_images, poses, gazes = _filter_subject_02(right_images, left_images, poses, gazes)
        if i == 10:
            right_images, left_images, poses, gazes = _filter_subject_10(right_images, left_images, poses, gazes)

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


def _filter_subject_02(right_images, left_images, poses, gazes):
    """
    Rejects records from day 21 for subject id 02.
    """
    mask = list()
    for i in range(len(right_images)):
        if i >= 15400 and i < 16323:
            mask.append(False)
        else:
            mask.append(True)

    return right_images[mask], left_images[mask], poses[mask], gazes[mask]


def _filter_subject_10(right_images, left_images, poses, gazes):
    """
    Rejects records from day 05 for subject id 10.
    """
    mask = list()
    for i in range(len(right_images)):
        if i >= 1030 and i < 1490:
            mask.append(False)
        else:
            mask.append(True)

    return right_images[mask], left_images[mask], poses[mask], gazes[mask]
