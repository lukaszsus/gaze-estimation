import os

import numpy as np
from tqdm import tqdm
from time import time
from application.utils import create_pipeline
from data_processing.utils import load_image_by_cv2, own_dataset_path_wrapper
from scripts.create_dataset.create_dataset_mpiigaze_processed_both_rgb import norm_coords
from settings import DATA_PATH


def create_own_dataset(dir_name: str):
    # screen resolution
    screen_size = _load_screen_resolution(dir_name)
    # screen_size = (768, 1366)

    # counters
    counter = 0     # count valid records
    broken_path_counter = 0
    no_face_detected_counter = 0

    # data to save
    right_image_list = list()
    left_image_list = list()
    pose_list = list()
    coords_list = list()

    # pipeline
    pipeline = create_pipeline(screen_size=screen_size)
    # pipeline_haarcascade_lbf = create_pipeline(face_detector="haarcascade", landmarks_detector="lbf")
    metadata = _load_metadata(dir_name)

    # time
    time_sum = 0

    for row in tqdm(metadata):
        file_path = _get_file_path(dir_name, row)
        coords = _get_coords(row)      # x, y
        coords = (coords[1], coords[0])      # y, x
        im = load_image_by_cv2(file_path)

        if im is None:
            broken_path_counter += 1
            continue

        start_time = time()

        data = pipeline.process(im)
        if data is None:
            no_face_detected_counter += 1
            continue

        elapsed = time() - start_time
        time_sum += elapsed

        counter += 1
        right_image_list.append(data["right_image"])
        left_image_list.append(data["left_image"])
        pose_list.append(data["pose"])
        coords_list.append(coords)

    print(f"Valid records: {counter}")
    print(f"Broken paths: {broken_path_counter}")
    print(f"No face detected: {no_face_detected_counter}")
    print(f"Average time: {time_sum / counter}")

    data = {"right_image": np.concatenate(right_image_list),
            "left_image": np.concatenate(left_image_list),
            "pose_landmarks": np.concatenate(pose_list),
            "coordinates": np.asarray(coords_list)}
    data["coordinates"] = norm_coords(data["coordinates"], screen_size)

    _save_own_dataset(data)


def _load_metadata(dir_name):
    file_path = os.path.join(dir_name, "metadata.txt")
    metadata = np.loadtxt(own_dataset_path_wrapper(file_path), dtype=str)
    return metadata


def _get_file_path(dir_name: str, row: list):
    file_path = row[0]
    file_path = file_path.replace("data/", "")
    file_path = own_dataset_path_wrapper(os.path.join(dir_name, file_path))
    return file_path


def _get_coords(row: list):
    x = float(row[1])
    y = float(row[2])
    return x, y


def _save_own_dataset(data: dict, dataset_name="own_dataset"):
    dataset_path = os.path.join(DATA_PATH, dataset_name)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    file_name = "p33.npz"
    file_path = os.path.join(dataset_path, file_name)
    with open(file_path, 'wb') as file:
        np.savez(file, right_image=data["right_image"], left_image=data["left_image"], pose=data["pose_landmarks"],
                 gaze=data["coordinates"])


def _load_screen_resolution(dir_name):
    """
    return height, width
    """
    res = np.loadtxt(own_dataset_path_wrapper(os.path.join(dir_name, "screen_resolution.txt")))
    return (res[1], res[0])


if __name__ == "__main__":
    create_own_dataset(dir_name="ja_laptop_rodzicow_20200523")

