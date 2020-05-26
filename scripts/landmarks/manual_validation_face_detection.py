import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from data_processing.utils import load_image_by_cv2, own_dataset_path_wrapper
from models.face_detectors.haarcascade_face_detector import HaarcascadeFaceDetector
from models.face_detectors.hog_face_detector import HogFaceDetector
from data_processing.own_dataset import _load_metadata
from settings import RESULTS_PATH
from utils.landmarks import visualize_faces

plt.ioff()


def experiment_detect_faces():
    dataset_size = 200
    out_dir = os.path.join(RESULTS_PATH, "manual_face_detection")
    os.makedirs(out_dir, exist_ok=True)

    haarcascade = HaarcascadeFaceDetector()
    hog = HogFaceDetector()

    metadata = _load_metadata()
    paths = get_random_subset_paths(metadata, dataset_size)

    for equal_hist in [False, True]:
        out_subdir = "equal_hist" if equal_hist else "raw_grayscale"
        out_subdir = os.path.join(out_dir, out_subdir)
        os.makedirs(out_subdir, exist_ok=True)

        for i, file_path in tqdm(enumerate(paths)):
            file_path = own_dataset_path_wrapper(file_path)
            image = load_image_by_cv2(file_path)
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            if equal_hist:
                image_gray = cv2.equalizeHist(image_gray)

            faces = haarcascade.detect(image_gray)
            visualize_faces(faces, image, color=(255, 0, 0), show=False)

            faces = hog.detect(image_gray)
            visualize_faces(faces, image, color=(0, 0, 255), show=False)

            path = os.path.join(out_subdir, str(i).zfill(4) + ".jpg")
            im = Image.fromarray(image)
            im.save(path)


def get_random_subset_paths(metadata, subset_size):
    if subset_size > len(metadata):
        raise ValueError("Value of subset_size has to be smaller than length of metadata.")

    paths = metadata[:, 0]
    paths = np.random.choice(paths, size=subset_size, replace=False)
    return paths


if __name__ == "__main__":
    experiment_detect_faces()