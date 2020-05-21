import cv2
import numpy as np
from pynput.mouse import Controller

from application.utils import create_pipeline


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

    pipeline = create_pipeline(model_name="own_mpiigaze")

    predictions = list()
    s, img = cam.read()
    if s:  # frame captured without any errors
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prediction = pipeline.predict(img)
        if prediction is not None:
            prediction = prediction.squeeze()
            print(mouse.position)
            print(prediction[1], prediction[0])
            #     predictions.append(prediction)
            #
            # if i % 10 == 9 and len(predictions) > 5:
            #     new_pos = np.concatenate(predictions, axis=0)
            #     new_pos = np.mean(new_pos, axis=0)
            #     mouse.position = (int(new_pos[1]), int(new_pos[0]))
            #     print(mouse.position)
            #     predictions = list()
