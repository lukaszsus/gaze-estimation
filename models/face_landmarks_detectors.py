import os
from urllib import request as urlreq

import cv2

from settings import PROJECT_PATH


def get_haarcascade_detector():
    """
    Face Bounding Box Detector
    """
    haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    _, haarcascade_file_name = os.path.splitext(haarcascade_url)
    model_dir = os.path.join(PROJECT_PATH, "models")
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

    _, LBFmodel_file_name = os.path.splitext(LBFmodel_url)
    model_dir = os.path.join(PROJECT_PATH, "models")
    model_path = os.path.join(model_dir, LBFmodel_file_name)

    if not os.path.isfile(model_path):
        urlreq.urlretrieve(LBFmodel_url, model_path)

    # create an instance of the Facial landmark Detector with the model
    landmark_detector = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(model_path)

    return landmark_detector