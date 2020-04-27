import os
import cv2
import numpy as np

from urllib import request as urlreq

from settings import PROJECT_PATH, DATA_PATH


def get_haarcascade_detector():
    """
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


def filter_lbf_model_landmarks(landmarks: list):
    """
    Function filters lbf model landmarks and returns only eyes and mouth's corners coordinates (6 landmarks).
    """
    if type(landmarks) == list:
        landmarks = np.asarray(landmarks)
    landmarks = landmarks.squeeze()
    return landmarks[[36, 39, 42, 45, 48, 54]]   # 36, 39, 42, 45, 48, 54