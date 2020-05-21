from tkinter import Tk

import numpy as np

from application.pipeline import Pipeline
from models.face_detectors.haarcascade_face_detector import HaarcascadeFaceDetector
from models.face_detectors.hog_face_detector import HogFaceDetector
from models.landmarks_detectors.kazemi_landmarks_detector import KazemiLandmarksDetector
from models.landmarks_detectors.lbf_landmarks_detector import LbfLandmarksDetector
from models.model_loaders import load_best_modal3_conv_net, load_modal3_conv_net_own_mpiigaze


def get_screen_size():
    window = Tk()
    width_px = window.winfo_screenwidth()
    height_px = window.winfo_screenheight()
    return height_px, width_px


def get_avg_camera_matrix():
    return np.asarray([[1.02794690e+03, 0.00000000e+00, 6.39687904e+02],
                       [0.00000000e+00, 1.03100212e+03, 3.60360146e+02],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


def create_pipeline(model_name="best", face_detector: str = "hog", landmarks_detector="kazemi"):
    eye_image_width = 60
    eye_image_height = 36

    camera_matrix = get_avg_camera_matrix()
    screen_size = get_screen_size()
    print(f"Screen size: {screen_size}")

    if face_detector == "hog":
        face_detector = HogFaceDetector()
    elif face_detector == "haarcascade":
        face_detector = HaarcascadeFaceDetector()

    if landmarks_detector == "kazemi":
        landmarks_detector = KazemiLandmarksDetector()
    elif landmarks_detector == "lbf":
        landmarks_detector = LbfLandmarksDetector()

    if model_name == "best":
        model = load_best_modal3_conv_net()
    elif model_name == "own_mpiigaze":
        model = load_modal3_conv_net_own_mpiigaze(test=False)

    pipeline = Pipeline(gaze_estimation_model=model,
                        face_detector=face_detector,
                        landmarks_detector=landmarks_detector,
                        eye_image_width=eye_image_width,
                        eye_image_height=eye_image_height,
                        camera_matrix=camera_matrix,
                        screen_size=screen_size)

    return pipeline