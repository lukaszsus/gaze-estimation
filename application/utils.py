from tkinter import Tk

import numpy as np


def get_screen_size():
    window = Tk()
    width_px = window.winfo_screenwidth()
    height_px = window.winfo_screenheight()
    return height_px, width_px


def get_avg_camera_matrix():
    return np.asarray([[1.02794690e+03, 0.00000000e+00, 6.39687904e+02],
                       [0.00000000e+00, 1.03100212e+03, 3.60360146e+02],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])