import os

import numpy as np
import cv2
from PIL import Image

from data_processing.utils import load_image_by_cv2, mpii_face_gaze_path_wrapper, mpiigaze_path_wrapper
from models.face_detectors.haarcascade_face_detector import HaarcascadeFaceDetector
from models.face_detectors.hog_face_detector import HogFaceDetector
from models.landmarks_detectors.kazemi_landmarks_detector import KazemiLandmarksDetector
from models.landmarks_detectors.landmarks_detector import EYES_LANDMARKS
from models.landmarks_detectors.lbf_landmarks_detector import LbfLandmarksDetector
from settings import FOR_THESIS_DIR
from utils.landmarks import visualize_landmarks_mpii_gaze_format, visualize_landmarks


def haarcascade_lbf(image):
    face_detector = HogFaceDetector()
    landmarks_detector = KazemiLandmarksDetector()

    # grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # detect face
    faces = face_detector.detect(image_gray)

    # Take only first face
    if len(faces) > 1:
        faces = faces[0].reshape(1, -1)
    elif len(faces) == 0:  # No face detected -> assume that face is on the full image
        return None

    # detect landmarks
    landmarks = landmarks_detector.detect(image_gray, faces)
    visualize_landmarks_mpii_gaze_format(landmarks[:, EYES_LANDMARKS, :], image, numbers=False)


if __name__ == "__main__":
    """
    Script prints all landmarks from MPII Gaze dataset on photo, to know which landmarks are they.
    """
    person_id_str = "14"
    day = "day04"
    im_file = "0008.jpg"
    im = load_image_by_cv2(mpii_face_gaze_path_wrapper(f"p{person_id_str}/{day}/{im_file}"))

    annotation = np.loadtxt(mpiigaze_path_wrapper(f"Data/Original/p{person_id_str}/{day}/annotation.txt"))
    landmarks = np.reshape(annotation[8 - 1, 0:24], (1, -1, 2))
    landmarks = landmarks.astype(np.int)

    visualize_landmarks_mpii_gaze_format(landmarks, im, numbers=True)

    # haarcascade_lbf(im)

    im = Image.fromarray(im)
    im.save(os.path.join(FOR_THESIS_DIR, "eye_landmarks.png"))
