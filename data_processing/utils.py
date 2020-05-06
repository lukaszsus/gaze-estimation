import os

import cv2

from settings import MPII_FACE_GAZE_PATH
from user_settings import MPIIGAZE_PATH, UT_MULTIVIEW_PATH


def mpiigaze_path_wrapper(path):
    path = os.path.join(MPIIGAZE_PATH, path)
    return path


def ut_multiview_path_wrapper(path):
    path = os.path.join(UT_MULTIVIEW_PATH, path)
    return path


def mpii_face_gaze_path_wrapper(path):
    path = os.path.join(MPII_FACE_GAZE_PATH, path)
    return path


def load_image_by_cv2(file_path):
    """
    Function load single image from file using cv2 method.
    :param file_path:
    :return: image as numpy array
    """
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

