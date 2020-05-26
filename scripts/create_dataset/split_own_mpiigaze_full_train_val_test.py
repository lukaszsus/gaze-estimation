import numpy as np
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from settings import DATA_PATH


def split_own_mpiigaze_full():
    val_split = 0.2
    src_mpiigaze_dir = "mpiigaze_both_landmarks_coords_full"
    src_own_dir = "own_dataset"

    mpiigaze_people_ids = list(range(0, 7)) + list(range(8, 15))
    own_people_ids = [24, 25] + list(range(30, 36))


    _split_train_val_test(src_mpiigaze_dir, "own_mpiigaze_full_train_val_test",
                          mpiigaze_people_ids, val_split=val_split)
    _split_train_val_test(src_own_dir, "own_mpiigaze_full_train_val_test",
                          own_people_ids, val_split=val_split)


def _split_train_val_test(src_dir, dest_dir, people_ids, val_split):
    dest_dir = os.path.join(DATA_PATH, dest_dir)
    train_val_dir = os.path.join(dest_dir, "train_val")
    test_dir = os.path.join(dest_dir, "test")
    os.makedirs(train_val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for person_id in tqdm(people_ids):
        file_name = "p" + str(person_id).zfill(2) + ".npz"
        src_file_path = os.path.join(DATA_PATH, src_dir, file_name)
        right_images, left_images, poses, gazes = _load_one_person(src_file_path)
        (right_eye_train, right_eye_test,
         left_eye_train, left_eye_test,
         headpose_train, headpose_test,
         y_train, y_test) = train_test_split(
            right_images, left_images, poses, gazes, test_size=val_split, random_state=42)

        train_val_file_path = os.path.join(train_val_dir, file_name)
        np.savez(train_val_file_path, right_image=right_eye_train,
                 left_image=left_eye_train, pose=headpose_train, gaze=y_train)

        test_file_path = os.path.join(test_dir, file_name)
        np.savez(test_file_path, right_image=right_eye_test,
                 left_image=left_eye_test, pose=headpose_test, gaze=y_test)


def _load_one_person(file_path):
    with np.load(file_path) as data:
        right_images = data['right_image']
        left_images = data['left_image']
        poses = data['pose']
        gazes = data['gaze']

    return right_images, left_images, poses, gazes


if __name__ == "__main__":
    split_own_mpiigaze_full()