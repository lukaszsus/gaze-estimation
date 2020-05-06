import os

import numpy as np

from application.pipeline import Pipeline
from application.utils import get_avg_camera_matrix, get_screen_size
from data_processing.utils import mpiigaze_path_wrapper, mpii_face_gaze_path_wrapper, load_image_by_cv2
from models.face_detectors.hog_face_detector import HogFaceDetector
from models.landmarks_detectors.kazemi_landmarks_detector import KazemiLandmarksDetector
from scripts.create_dataset.create_dataset_mpiigaze_processed_both_rgb import load_camera_matrix, load_screen_size
from settings import PHOTO_TAKER_DATA_PATH


def read_metadata():
    file_path = os.path.join(PHOTO_TAKER_DATA_PATH, "metadata.txt")
    return np.loadtxt(file_path, dtype='U100')


if __name__ == "__main__":
    eye_image_width = 60
    eye_image_height = 36

    camera_matrix = get_avg_camera_matrix()
    screen_size = get_screen_size()

    face_detector = HogFaceDetector()
    landmarks_detector = KazemiLandmarksDetector()

    pipeline = Pipeline(face_detector=face_detector,
                        landmarks_detector=landmarks_detector,
                        eye_image_width=eye_image_width,
                        eye_image_height=eye_image_height,
                        camera_matrix=camera_matrix,
                        screen_size=screen_size)

    metadata = read_metadata()
    for row in metadata:
        file_path = os.path.join(PHOTO_TAKER_DATA_PATH, row[0])
        coords = row[1:].astype(np.int)
