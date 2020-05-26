import os

import numpy as np

from data_processing.utils import own_dataset_path_wrapper


def _load_screen_resolution(dir_name):
    """
    return height, width
    """
    res = np.loadtxt(own_dataset_path_wrapper(os.path.join(dir_name, "screen_resolution.txt")))
    return (res[1], res[0])


def _load_metadata(dir_name):
    file_path = os.path.join(dir_name, "metadata.txt")
    metadata = np.loadtxt(own_dataset_path_wrapper(file_path), dtype=str)
    return metadata


def _get_file_path(dir_name: str, row: list):
    file_path = row[0]
    file_path = file_path.replace("data/", "")
    file_path = own_dataset_path_wrapper(os.path.join(dir_name, file_path))
    return file_path


def _get_coords(row: list):
    x = float(row[1])
    y = float(row[2])
    return x, y