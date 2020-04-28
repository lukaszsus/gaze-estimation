import os
import cv2
import numpy as np

from models.face_detectors.face_detector import FaceDetector
from settings import DATA_PATH
from urllib import request as urlreq


class HaarcascadeFaceDetector(FaceDetector):
    """
    Face Bounding Box Detector using Haarcascade method implemented with OpenCV.
    """
    def __init__(self):
        haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
        haarcascade_file_name = os.path.basename(haarcascade_url)
        model_dir = os.path.join(DATA_PATH, "models", "face_landmarks")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, haarcascade_file_name)

        if not os.path.exists(model_path):
            urlreq.urlretrieve(haarcascade_url, model_path)

        # create an instance of the Face Detection Cascade Classifier
        self.detector = cv2.CascadeClassifier(model_path)

    def detect(self, image) -> np.ndarray:
        return self.detector.detectMultiScale(image)

    def __repr__(self):
        return "Haarcascade"

