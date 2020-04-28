import os
import numpy as np
import cv2

from urllib import request as urlreq

from models.landmarks_detectors.landmarks_detector import LandmarksDetector
from settings import DATA_PATH


class LbfLandmarksDetector(LandmarksDetector):
    def __init__(self):
        """
        Facial Landmark Detection
        """
        LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

        LBFmodel_file_name = os.path.basename(LBFmodel_url)
        model_dir = os.path.join(DATA_PATH, "models", "face_landmarks")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, LBFmodel_file_name)

        if not os.path.isfile(model_path):
            urlreq.urlretrieve(LBFmodel_url, model_path)

        # create an instance of the Facial landmark Detector with the model
        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel(model_path)

    def detect(self, image, faces) -> np.ndarray:
        _, landmarks = self.landmark_detector.fit(image, faces)
        return filter_lbf_model_landmarks(landmarks)

    def __repr__(self):
        return "LBF"


def filter_lbf_model_landmarks(landmarks: list):
    """
    Function filters lbf model landmarks and returns only eyes and mouth's corners coordinates (6 landmarks).
    """
    if type(landmarks) == list:
        landmarks = np.asarray(landmarks)
    landmarks = landmarks.squeeze()
    return landmarks[[36, 39, 42, 45, 48, 54]]   # 36, 39, 42, 45, 48, 54