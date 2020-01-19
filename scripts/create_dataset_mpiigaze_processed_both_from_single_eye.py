import os

import numpy as np
from data_processing.utils import mpiigaze_path_wrapper
from scripts.create_dataset_mpiigaze_processed_both_rgb import create_dirs, parse_mpiigaze, get_all_days, \
    get_all_jpg_files
from scripts.create_dataset_mpiigaze_processed_one_eye import get_all_days_ids, get_all_images_ids_for_day
from settings import DATA_PATH
from utils.metrics import convert_to_unit_vector


def vec2angle(vectors_as_rows):
    x = vectors_as_rows[:, 0]
    y = vectors_as_rows[:, 1]
    z = vectors_as_rows[:, 2]
    yaw = np.arctan2(-x, -z)
    pitch = np.arcsin(-y)
    return np.stack([yaw, pitch], axis=1)


def resultant_angles(angles_1, angles_2):
    """
    Computes mean angle error between two vectors.
    Uses tensorflow.
    """
    x_1, y_1, z_1 = convert_to_unit_vector(angles_1)
    x_2, y_2, z_2 = convert_to_unit_vector(angles_2)
    vector = np.stack([x_1 + x_2, y_1 + y_2, z_1 + z_2], axis=1)
    vector = vector / np.linalg.norm(vector, axis=1).reshape((-1, 1))
    return vec2angle(vector)


def prepare_data_for_one_person(person_id: int, limit: int = 3000):
    counter = 0
    data = dict()
    data["right_image"] = list()
    data["left_image"] = list()
    data["pose"] = list()
    data["gaze"] = list()

    def prepare_output(data: dict):
        data["right_image"] = np.stack(data["right_image"])
        data["left_image"] = np.stack(data["left_image"])
        data["pose"] = np.array(data["pose"])
        resultant_pose = resultant_angles(data["pose"][:, 0:2], data["pose"][:, 2:4])
        data["pose"] = np.concatenate([data["pose"], resultant_pose], axis=1)
        data["gaze"] = np.array(data["gaze"])
        data["gaze"] = resultant_angles(data["gaze"][:, 0:2], data["gaze"][:, 2:4])
        return data

    days_ids = get_all_days_ids(person_id)
    for day_id in days_ids:
        im_ids = get_all_images_ids_for_day(person_id, day_id)
        for im_id in im_ids:
            right_eye, left_eye = parse_mpiigaze(person_id, day_id, im_id)
            data["right_image"].append(right_eye["img"])
            data["left_image"].append(left_eye["img"])
            pose_angles = [right_eye["headpose_theta"], right_eye["headpose_phi"],
                           left_eye["headpose_theta"], left_eye["headpose_phi"]]
            data["pose"].append(pose_angles)
            gaze_angles = [right_eye["gaze_theta"], right_eye["gaze_phi"],
                           left_eye["gaze_theta"], left_eye["gaze_phi"]]
            data["gaze"].append(gaze_angles)
            counter += 1
            if counter % 100 == 0:
                print(counter)
            if counter == limit:
                return prepare_output(data)
    return prepare_output(data)


def create_mpiigaze_full_transformation_both(dataset_name, limit):
    """
    Creates dataset similiar to mpiigaze_processed_one_eye but tries to combine both eyes.
    """
    dataset_path = os.path.join(DATA_PATH, dataset_name)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    for person_id in range(15):
        print(f"person id: {person_id}")
        data = prepare_data_for_one_person(person_id, limit=limit)
        file_name = f"p{str(person_id).zfill(2)}.npz"
        file_path = os.path.join(dataset_path, file_name)
        with open(file_path, 'wb') as file:
            np.savez(file, right_image=data["right_image"], left_image=data["left_image"], pose=data["pose"],
                     gaze=data["gaze"])


if __name__ == '__main__':
    create_mpiigaze_full_transformation_both(dataset_name="mpiigaze_both_like_hysts", limit=3000)
