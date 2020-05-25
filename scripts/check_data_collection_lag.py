import os
import numpy as np
from datetime import datetime
from settings import OWN_DATASET_PATH


def count_lag(dir_path: str):
    metadata_path = os.path.join(dir_path, "metadata.txt")
    logs_path = os.path.join(dir_path, "logs.txt")

    metadata = np.loadtxt(metadata_path, dtype=str)
    metadata_time = get_time_from_metadata(metadata)

    logs_time = get_time_from_logs(logs_path)

    print(np.mean(logs_time - metadata_time))


def get_time_from_metadata(metadata):
    file_names = metadata[:, 0]
    timestamps = list()
    for file_name in file_names:
        file_name = file_name.split("/")[2]
        timestamp_str = file_name.split(".")[0]      # remove file extension
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d-t-%H-%M-%S-%f')
        timestamps.append(timestamp)

    return np.asarray(timestamps)


def get_time_from_logs(logs_path: str):
    with open(logs_path, 'r') as file:
        lines = file.readlines()

    timestamps = list()
    for line in lines:
        timestamp_str = line.split()[0:2]
        timestamp_str = timestamp_str[0] + "-" + timestamp_str[1]
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d-%H:%M:%S,%f:')
        timestamps.append(timestamp)

    return np.asarray(timestamps)


if __name__ == "__main__":
    print("Linux:")
    dir_path = os.path.join(OWN_DATASET_PATH, "Marta_images")
    count_lag(dir_path)

    print("Windows:")
    dir_path = os.path.join(OWN_DATASET_PATH, "ja_laptop_rodzicow_20200523")
    count_lag(dir_path)

    print("Windows best:")
    dir_path = os.path.join(OWN_DATASET_PATH, "tata_20200523_laptop_rodzicow")
    count_lag(dir_path)
