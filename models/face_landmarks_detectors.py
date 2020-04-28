import os
import cv2

from urllib import request as urlreq

from settings import DATA_PATH


"""
File deprecated. Instead of that use classes defined in models/face_detectors and models/landmarks_detectors.
"""


def get_haarcascade_detector():
    """
    DEPRECATED. Instead, use class HaarcascadeFaceDetector defined in models/face_detectors/haarcascade_face_detector.py.
    Face Bounding Box Detector
    """
    haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    haarcascade_file_name = os.path.basename(haarcascade_url)
    model_dir = os.path.join(DATA_PATH, "models", "face_landmarks")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, haarcascade_file_name)

    if not os.path.exists(model_path):
        urlreq.urlretrieve(haarcascade_url, model_path)

    # create an instance of the Face Detection Cascade Classifier
    detector = cv2.CascadeClassifier(model_path)

    return detector


def get_lbf_model():
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
    landmark_detector = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(model_path)

    return landmark_detector
