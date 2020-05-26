import os
import numpy as np
import time
from tkinter import Canvas, Tk, mainloop

import cv2
from pynput import keyboard
from pynput.mouse import Controller
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from tqdm import tqdm

from application.utils import create_pipeline, get_screen_size
from data_processing.own_dataset import _load_screen_resolution, _load_metadata, _get_file_path, _get_coords
from data_processing.utils import load_image_by_cv2
from settings import DATA_PATH

break_program = False


def set_max_camera_res(cam):
    HIGH_VALUE = 10000

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, HIGH_VALUE)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HIGH_VALUE)

    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)
    return cam


def on_press(key):
    global break_program
    print(key)
    if key == keyboard.Key.end:     # or key.char == 'q':
        break_program = True
        return False


def start_app_mouse_controller():
    """
    Exponentially weighted moving average:
    v(t) = beta * v(t-1) + (1-beta)*theta(t)

    average over:
    1/(1-beta) numbers
    """
    cam = cv2.VideoCapture(0)
    cam = set_max_camera_res(cam)
    mouse = Controller()

    screen_size = get_screen_size()
    pipeline = create_pipeline(model_name="best", screen_size=screen_size)
    # pipeline = create_pipeline(model_name="modal3_conv_net_mean_camera_matrix", screen_size=screen_size)
    # pipeline = create_pipeline(model_name="modal3_conv_net_full_0.h5", screen_size=screen_size)

    pred_x = float(screen_size[1] / 2)
    pred_y = float(screen_size[0] / 2)

    # keyboard listener

    with keyboard.Listener(on_press=on_press) as listener:
        while not break_program:
            s, img = cam.read()
            if s:  # frame captured without any errors
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                prediction = pipeline.predict(img)
                if prediction is not None:
                    prediction = prediction.squeeze()
                    pred_x = 0.85 * pred_x + 0.15 * prediction[1]
                    pred_y = 0.85 * pred_y + 0.15 * prediction[0]
                    mouse.position = (int(pred_x), int(pred_y))
        listener.join()


def start_app_mouse_controller_calibration():
    """
    Exponentially weighted moving average:
    v(t) = beta * v(t-1) + (1-beta)*theta(t)

    average over:
    1/(1-beta) numbers
    """
    cam = cv2.VideoCapture(0)
    cam = set_max_camera_res(cam)
    mouse = Controller()

    screen_size = get_screen_size()
    pipeline = create_pipeline(model_name="best", screen_size=screen_size)
    # pipeline = create_pipeline(model_name="modal3_conv_net_full_0.h5", screen_size=screen_size)

    coordinates, predictions = _prepare_data_for_calibration(pipeline, dir_name="20200525")
    regressor = _fit_calibration(predictions, coordinates)

    pred_x = float(screen_size[1] / 2)
    pred_y = float(screen_size[0] / 2)

    # keyboard listener

    with keyboard.Listener(on_press=on_press) as listener:
        while not break_program:
            s, img = cam.read()
            if s:  # frame captured without any errors
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                prediction = pipeline.predict(img)
                if prediction is not None:
                    pred = regressor.predict(prediction)
                    pred_x = 0.85 * pred_x + 0.15 * pred[0, 1]
                    pred_y = 0.85 * pred_y + 0.15 * pred[0, 0]
                    mouse.position = (int(pred_x), int(pred_y))
        listener.join()


def _prepare_data_for_calibration(pipeline, dir_name):
    dir_path = os.path.join(DATA_PATH, "application", "calibration", dir_name)
    metadata = _load_metadata(dir_path)
    coords_list = list()
    predictions = list()
    for row in tqdm(metadata):
        file_path = _get_file_path(dir_path, row)
        coords = _get_coords(row)      # x, y
        coords = (coords[1], coords[0])      # y, x

        if not os.path.exists(file_path):
            continue
        im = load_image_by_cv2(file_path)

        if im is None:
            continue

        prediction = pipeline.predict(im)
        predictions.append(prediction.squeeze())
        coords_list.append(coords)
    return np.asarray(coords_list), np.asarray(predictions)


def _fit_calibration(X, y):
    clf = LinearRegression(n_jobs=-1)
    # clf = SVR(C=0.01, epsilon=0.02)
    clf = clf.fit(X, y)
    return clf


if __name__ == "__main__":
    start_app_mouse_controller()
    # start_app_mouse_controller_calibration()