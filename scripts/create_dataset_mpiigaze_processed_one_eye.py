import os

import numpy as np
from data_processing.utils import mpiigaze_path_wrapper
from scripts.create_dataset_mpiigaze_processed_both_rgb import create_dirs, parse_mpiigaze, get_all_days, \
    get_all_jpg_files
from settings import DATA_PATH


def get_all_days_ids(person_id):
    """
    Function retrieves days ids from strings like 'day{xx}'.
    :days_list_str: list containing days as str: day{xx} where xx is id
    """
    person_id_str = f"p{str(person_id).zfill(2)}"
    days_list_str = get_all_days(path=mpiigaze_path_wrapper(f"Data/Original/{person_id_str}/"))
    ids = [int(day.replace("day", "")) for day in days_list_str]
    return ids


def get_all_images_ids_for_day(person_id, day_id):
    person_id_str = f"p{str(person_id).zfill(2)}"
    day = f"day{str(day_id).zfill(2)}"
    im_filenames = get_all_jpg_files(mpiigaze_path_wrapper(f"Data/Original/{person_id_str}/{day}/"))
    ids = [int(filename.replace(".jpg", "")) for filename in im_filenames]
    return ids


def prepare_data_for_one_person(person_id: int, limit: int = 3000):
    counter = 0
    data = dict()
    data["image"] = list()
    data["pose"] = list()
    data["gaze"] = list()

    def prepare_output(data: dict):
        data["image"] = np.stack(data["image"])
        data["pose"] = np.stack(data["pose"])
        data["gaze"] = np.stack(data["gaze"])
        return data

    days_ids = get_all_days_ids(person_id)
    for day_id in days_ids:
        im_ids = get_all_images_ids_for_day(person_id, day_id)
        for im_id in im_ids:
            eyes_data = parse_mpiigaze(person_id, day_id, im_id)
            for eye in eyes_data:
                data["image"].append(eye["img"])
                data["pose"].append([eye["headpose_theta"], eye["headpose_phi"]])
                data["gaze"].append([eye["gaze_theta"], eye["gaze_phi"]])
                counter += 1
                if counter % 100 == 0:
                    print(counter)
                if counter == limit:
                    return prepare_output(data)
    return prepare_output(data)


def create_mpiigaze_full_transformation_one_eye(dataset_name):
    """
    Creates the same dataset the same as hysts or at least in the same format.
    One eye, 2 angles for headpose and, 2 angles for gaze.
    It differs from hysts version because this function does not mirror
    right eye into left eye (or otherwise). So model must be more universal for this dataset.
    """
    dataset_path = os.path.join(DATA_PATH, dataset_name)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    for person_id in range(15):
        print(f"person id: {person_id}")
        data = prepare_data_for_one_person(person_id, limit=3000)
        file_name = f"p{str(person_id).zfill(2)}.npz"
        file_path = os.path.join(dataset_path, file_name)
        with open(file_path, 'wb') as file:
            np.savez(file, image=data["image"], pose=data["pose"], gaze=data["gaze"])


if __name__ == '__main__':
    create_mpiigaze_full_transformation_one_eye(dataset_name="mpiigaze_processed_one_eye_like_hysts")