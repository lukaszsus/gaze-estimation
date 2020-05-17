import cv2
import numpy as np
from pynput.mouse import Controller

from application.pipeline import Pipeline
from application.utils import get_screen_size, get_avg_camera_matrix
from models.face_detectors.hog_face_detector import HogFaceDetector
from models.landmarks_detectors.kazemi_landmarks_detector import KazemiLandmarksDetector


def set_max_camera_res(cam):
    HIGH_VALUE = 10000

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, HIGH_VALUE)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HIGH_VALUE)

    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)
    return cam


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    cam = set_max_camera_res(cam)
    mouse = Controller()

    eye_image_width = 60
    eye_image_height = 36

    camera_matrix = get_avg_camera_matrix()
    screen_size = get_screen_size()

    face_detector = HogFaceDetector()
    landmarks_detector = KazemiLandmarksDetector()

    pipeline = Pipeline(face_detector=face_detector,
                        landmarks_detector=landmarks_detector,
                        eye_image_width=eye_image_width,
                        eye_image_height=eye_image_height,
                        camera_matrix=camera_matrix,
                        screen_size=screen_size)

    predictions = list()
    for i in range(200):
        s, img = cam.read()
        if s:  # frame captured without any errors
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            prediction = pipeline.predict(img)
            if prediction is not None:
                predictions.append(prediction)

            if i % 10 == 9 and len(predictions) > 5:
                new_pos = np.concatenate(predictions, axis=0)
                new_pos = np.mean(new_pos, axis=0)
                mouse.position = (int(new_pos[1]), int(new_pos[0]))
                print(mouse.position)
                predictions = list()

            # print(mouse.position)
