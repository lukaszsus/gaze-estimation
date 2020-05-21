import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader.mpiigaze_both_from_single import _load_all_people_reject_suspicious
from data_processing.utils import load_image_by_cv2, mpiigaze_path_wrapper
from scripts.create_dataset.create_dataset_mpiigaze_processed_both_rgb import load_screen_size, load_face_model
from scripts.create_dataset.create_dataset_mpiigaze_processed_one_eye import get_all_days_ids, \
    get_all_images_ids_for_day


def get_mpiigaze_data():
    df_untrusted = pd.DataFrame(columns=["person_id", "day", "counter"])

    image_counter = dict()
    x_coords = list()
    y_coords = list()

    for person_id in range(15):
        person_id_str = str(person_id).zfill(2)
        screen_size = load_screen_size(path=f"Data/Original/p{person_id_str}/Calibration/screenSize.mat")

        days_ids = get_all_days_ids(person_id)
        for day_id in days_ids:
            day_str = str(day_id).zfill(2)
            annotation = np.loadtxt(
                mpiigaze_path_wrapper(f"Data/Original/p{person_id_str}/day{day_str}/annotation.txt"))
            if len(annotation.shape) == 1:
                annotation = annotation.reshape((1, -1))

            im_ids = get_all_images_ids_for_day(person_id, day_id)
            for im_id in im_ids:
                coordinates = np.array([annotation[im_id - 1, 25], annotation[im_id - 1, 24]])
                if float(coordinates[0]) / screen_size[0] > 1.0 or float(coordinates[1]) / screen_size[1] > 1.0:
                    df_untrusted = df_untrusted.append([{"person_id": person_id_str,
                                                         "day": day_str,
                                                         "counter": 1}], ignore_index=True)
                    continue
                if person_id in image_counter:
                    image_counter[person_id] += 1
                else:
                    image_counter[person_id] = 1
                x_coords.append(float(coordinates[0]) / screen_size[0])
                y_coords.append(float(coordinates[1]) / screen_size[1])

    return image_counter, x_coords, y_coords, df_untrusted


def get_processed_data_own_mpiigaze():
    dataset_name = "own_mpiigaze"
    grayscale = False
    all_subjects = list(range(0, 7)) + list(range(8, 15)) + ["23"]

    test_subject_ids = None
    right_images, left_images, poses, gazes, subject_ids = _load_all_people_reject_suspicious(dataset_name,
                                                                                              grayscale,
                                                                                              subjects=all_subjects)
    return right_images, left_images, poses, gazes, subject_ids


if __name__ == "__main__":
    image_counter, x_coords, y_coords = get_mpiigaze_data()
    print(image_counter)
    print(x_coords)
    print(y_coords)
