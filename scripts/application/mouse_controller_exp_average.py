import time
from tkinter import Canvas, Tk, mainloop

import cv2
from pynput import keyboard
from pynput.mouse import Controller

from application.utils import create_pipeline, get_screen_size

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
    pipeline = create_pipeline(model_name="modal3_conv_net_mean_camera_matrix", screen_size=screen_size)

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
                    pred_x = 0.90 * pred_x + 0.1 * prediction[1]
                    pred_y = 0.90 * pred_y + 0.1 * prediction[0]
                    mouse.position = (int(pred_x), int(pred_y))
        listener.join()


if __name__ == "__main__":
    start_app_mouse_controller()