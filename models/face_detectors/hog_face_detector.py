import time

import dlib
import numpy as np

from imutils import face_utils
from models.face_detectors.face_detector import FaceDetector


class HogFaceDetector(FaceDetector):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image) -> np.ndarray:
        rects = self.detector(image)
        seq = map(face_utils.rect_to_bb, rects)
        # seq = map(lambda rect: [rect.left(), rect.top(), rect.width(), rect.height()], rects)
        faces = np.asarray(list(seq))
        return faces

    def __repr__(self):
        return "HOG"
