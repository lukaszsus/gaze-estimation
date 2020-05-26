import os
import numpy as np
from tqdm import tqdm

from data_processing.own_dataset import _load_screen_resolution, _load_metadata, _get_file_path, _get_coords
from settings import OWN_DATASET_PATH, DATA_PATH

data_names_map = {'ja_20200519': 24, 'ja_20200520': 25, 'ja_20200523': 30, 'ja_laptop_rodzicow_20200523': 33,
                  'mama_20200523': 32, 'mma_images': 34, 'tata_20200523_laptop_rodzicow': 31,
                  'tata_20200524_moj_laptop': 35}


def get_own_dataset_data():
    image_counter = dict()
    face_detected_counter = dict()
    x_coords = list()
    y_coords = list()

    dir_names = _get_dir_names()

    for dir_name in dir_names:
        dir_path = os.path.join(OWN_DATASET_PATH, dir_name)

        screen_size = _load_screen_resolution(dir_path)
        metadata = _load_metadata(dir_path)
        counter = 0

        for row in tqdm(metadata):
            file_path = _get_file_path(dir_name, row)
            x, y = _get_coords(row)  # x, y

            if not os.path.exists(file_path):
                continue

            counter += 1
            x_coords.append(x / screen_size[1])
            y_coords.append(y / screen_size[0])

        face_detected_counter[dir_name] = get_number_after_processing(data_names_map[dir_name])
        image_counter[dir_name] = counter

    return image_counter, face_detected_counter, x_coords, y_coords


def _get_dir_names():
    dir_names = os.listdir(OWN_DATASET_PATH)
    dir_names = [dir_name for dir_name in dir_names
                 if os.path.isdir(os.path.join(OWN_DATASET_PATH, dir_name)) and dir_name != "data_old"]
    return dir_names


def get_number_after_processing(person_id: int):
    file_path = os.path.join(DATA_PATH, "own_dataset", "p" + str(person_id).zfill(2) + ".npz")
    data = np.load(file_path)
    length = data["pose"].shape[0]
    return length


if __name__ == "__main__":
    image_counter, face_detected_counter, x_coords, y_coords = get_own_dataset_data()
    print(image_counter)
    print(face_detected_counter)
    print(x_coords)
    print(y_coords)