import numpy as np
from abc import ABC, abstractmethod


class LandmarksDetector(ABC):
    @abstractmethod
    def detect(self, image, faces) -> np.ndarray:
        pass


MOUTH_EYES_CORNERS = [36, 39, 42, 45, 48, 54]
MOUTH_EYES_CORNERS_HEAD_POSE = [45, 42, 36, 39, 54, 48]
HEAD_POSE = [30, 8, 36, 45, 48, 54]
EYES_LANDMARKS = list(range(36, 48))


def filter_landmarks(landmarks: list, indices=None):
    """
    Function filters lbf model landmarks and returns only eyes and mouth's corners coordinates (6 landmarks).
    """
    if type(landmarks) == list:
        landmarks = np.asarray(landmarks)
    landmarks = landmarks.squeeze()
    return landmarks[indices]