import os
import numpy as np

from data_loader.mpiigaze_processed_loader import prepare_dataset, _prepare_images, _prepare_headposes, _prepare_gaze
from settings import DATA_PATH
from utils.configs import USE_FLOAT64


def load_test_mpii_face_gaze(dataset_name, person_id, val_split=0.2, batch_size=128,
                                                 grayscale=True, all_subjects=None):
    if all_subjects is None:
        all_subjects = list(range(0, 7)) + list(range(8, 15))

    right_images, left_images, poses, gazes, test_subject_ids = load_all_full_data(dataset_name, grayscale)
    indices = np.isin(test_subject_ids, all_subjects)

    right_images = right_images[indices]
    left_images = left_images[indices]
    poses = poses[indices]
    gazes = gazes[indices]
    test_subject_ids = test_subject_ids[indices]

    # don't shuffle test dataset to have consistent outcomes
    test_dataset = prepare_dataset((right_images, left_images, poses), gazes, batch_size, shuffle=False)

    return test_dataset, test_subject_ids


def load_all_full_data(dataset_name: str, grayscale: bool):
    path = f"{dataset_name}/all_full.npz"
    path = os.path.join(os.path.join(DATA_PATH, path))
    with np.load(path) as fin:
        right_images = _prepare_images(fin['right_image'], grayscale)
        left_images = _prepare_images(fin['left_image'], grayscale)
        poses = _prepare_headposes(fin['pose'], normalize=False)
        gazes = _prepare_gaze(fin['gaze'])
        people_ids = fin['person_id']

    if USE_FLOAT64:
        return right_images.astype(np.float64), left_images.astype(np.float64), poses.astype(np.float64), gazes.astype(
            np.float64)
    return right_images, left_images, poses, gazes, people_ids
