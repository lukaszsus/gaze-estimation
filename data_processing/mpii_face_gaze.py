import os
import numpy as np

from data_processing.utils import load_image_by_cv2
from settings import MPII_FACE_GAZE_PATH


def extract_landmarks_from_metadata_file(person_id: int):
    person_str = "p" + str(person_id).zfill(2)
    metadata_file_path = os.path.join(MPII_FACE_GAZE_PATH, person_str)
    metadata_file_path = os.path.join(metadata_file_path, person_str + ".txt")
    with open(metadata_file_path, "r") as file:
        lines = file.readlines()

    file_names = list()
    landmarks = list()
    for line in lines:
        data_line = line.split()
        file_names.append(data_line[0])
        image_landmarks = list()
        for i in range(3, 15, 2):
            image_landmarks.append([data_line[i], data_line[i + 1]])
        landmarks.append(image_landmarks)

    return file_names, landmarks


def load_mpii_face_gaze_image(person_id: int, person_image_path: str):
    person_str = "p" + str(person_id).zfill(2)
    full_file_path = os.path.join(MPII_FACE_GAZE_PATH, person_str, person_image_path)
    if os.path.exists(full_file_path):
        im = load_image_by_cv2(full_file_path)
    else:
        print("Image file does not exist.")
        im = None
    return im
