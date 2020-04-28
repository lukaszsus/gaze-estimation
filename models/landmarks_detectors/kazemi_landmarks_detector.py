import os
import cv2
import dlib
import numpy as np

from models.landmarks_detectors.landmarks_detector import LandmarksDetector
from settings import DATA_PATH
from urllib import request as urlreq


class KazemiLandmarksDetector(LandmarksDetector):
    def __init__(self):
        # model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        model_url = "https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat?raw=true"
        model_file_name = os.path.basename(model_url)
        model_dir = os.path.join(DATA_PATH, "models", "face_landmarks")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_file_name)

        if not os.path.exists(model_path):
            urlreq.urlretrieve(model_url, model_path)

        self.predictor = dlib.shape_predictor(model_path)

    def detect(self, image, faces) -> np.ndarray:
        landmarks = list()
        faces = list(map(lambda x: [x[0], x[1], x[0] + x[2], x[1] + x[3]], faces))
        for face in faces:
            shape = self.predictor(image, dlib.rectangle(face[0], face[1], face[2], face[3]))
            shape = list(map(shape.part, [36, 39, 42, 45, 48, 54]))
            shape = list(map(lambda p: [p.x, p.y], shape))
            landmarks.append(shape)
        return np.asarray(landmarks)

    def __repr__(self):
        return "Kazemi"

