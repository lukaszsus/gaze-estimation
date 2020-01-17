# coding: utf-8

import os
import numpy as np
from sklearn.model_selection import train_test_split

from data_loader.mpiigaze_processed_loader import prepare_dataset_generator
from user_settings import DATA_PATH


def _load_one_person(person):
    """
    :person: str or int - id of person
    """
    path = f"hysts_mpiigazed_processed/p{str(person).zfill(2)}.npz"
    path = os.path.join(os.path.join(DATA_PATH, path))
    with np.load(path) as fin:
        images = fin['image']
        poses = fin['pose']
        gazes = fin['gaze']
    return images, poses, gazes


def load_hysts_mpiigaze_train_test_ds_generator(person, val_split=0.2, batch_size=128):
    images, poses, gazes = _load_one_person(person)
    (eye_train, eye_test,
     headpose_train, headpose_test,
     y_train, y_test) = train_test_split(images, poses, gazes, test_size=val_split, random_state=42)
    train_dataset = prepare_dataset_generator((eye_train, headpose_train), y_train, batch_size)
    test_dataset = prepare_dataset_generator((eye_test, headpose_test), y_test, batch_size)
    return train_dataset, test_dataset
