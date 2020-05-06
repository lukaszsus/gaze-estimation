import os

import cv2
import numpy as np
from data_processing.utils import mpiigaze_path_wrapper, load_image_by_cv2
from scripts.create_dataset.create_dataset_mpiigaze_processed_both_from_single_eye import resultant_angles
from scripts.create_dataset.create_dataset_mpiigaze_processed_both_rgb import load_face_model, load_camera_matrix, get_img_gaze_headpose_per_eye, load_screen_size, \
    norm_coords, norm_landmarks
from scripts.create_dataset.create_dataset_mpiigaze_processed_one_eye import get_all_days_ids, get_all_images_ids_for_day
from settings import DATA_PATH


def parse_mpiigaze_landmark_coords(person_id: int, day: int, img_n: int, eye_image_width=60, eye_image_height=36):
    face_model = load_face_model()

    person_id_str = str(person_id).zfill(2)
    day_str = str(day).zfill(2)
    img_n_str = str(img_n).zfill(4)
    im = load_image_by_cv2(mpiigaze_path_wrapper(f"Data/Original/p{person_id_str}/day{day_str}/{img_n_str}.jpg"))
    annotation = np.loadtxt(mpiigaze_path_wrapper(f"Data/Original/p{person_id_str}/day{day_str}/annotation.txt"))
    camera_matrix = load_camera_matrix(path=f"Data/Original/p{person_id_str}/Calibration/Camera.mat")

    im_height, im_width, _ = im.shape

    headpose_hr = np.reshape(annotation[img_n - 1, 29:32], (1, -1))
    headpose_ht = np.reshape(annotation[img_n - 1, 32:35], (1, -1))
    h_r, _ = cv2.Rodrigues(headpose_hr)
    fc = np.dot(h_r, face_model)
    fc = fc + np.reshape(headpose_ht, (-1, 1))

    gaze_target = annotation[img_n - 1, 26:29]
    gaze_target = np.reshape(gaze_target, (-1, 1))

    right_eye_center = 0.5 * (fc[:, 0] + fc[:, 1])
    left_eye_center = 0.5 * (fc[:, 2] + fc[:, 3])

    right_eye = get_img_gaze_headpose_per_eye(im, right_eye_center, h_r, gaze_target,
                                              eye_image_width, eye_image_height, camera_matrix)
    left_eye = get_img_gaze_headpose_per_eye(im, left_eye_center, h_r, gaze_target,
                                             eye_image_width, eye_image_height, camera_matrix)

    landmarks = np.reshape(annotation[img_n - 1, 0:24], (1, -1))
    landmarks = norm_landmarks(landmarks, height=im_height, width=im_width)
    coordinates = np.array([annotation[img_n - 1, 25], annotation[img_n - 1, 24]])

    return right_eye, left_eye, landmarks, coordinates


def prepare_data_for_one_person(person_id: int, limit: int = 3000):
    counter = 0
    data = dict()
    data["right_image"] = list()
    data["left_image"] = list()
    data["pose"] = list()
    data["landmarks"] = list()
    data["coordinates"] = list()

    person_id_str = str(person_id).zfill(2)
    screen_size = load_screen_size(path=f"Data/Original/p{person_id_str}/Calibration/screenSize.mat")

    def prepare_output(data: dict):
        data["right_image"] = np.stack(data["right_image"])
        data["left_image"] = np.stack(data["left_image"])
        data["pose"] = np.array(data["pose"])
        resultant_pose = resultant_angles(data["pose"][:, 0:2], data["pose"][:, 2:4])
        data["pose"] = np.concatenate([data["pose"], resultant_pose], axis=1)
        data["landmarks"] = np.stack(data["landmarks"])
        data["pose_landmarks"] = np.concatenate([data["pose"], data["landmarks"]], axis=1)
        data["coordinates"] = np.array(data["coordinates"])
        data["coordinates"] = norm_coords(data["coordinates"], screen_size)
        return data

    days_ids = get_all_days_ids(person_id)
    for day_id in days_ids:
        im_ids = get_all_images_ids_for_day(person_id, day_id)
        for im_id in im_ids:
            right_eye, left_eye, landmarks, coords = parse_mpiigaze_landmark_coords(person_id, day_id, im_id)
            data["right_image"].append(right_eye["img"])
            data["left_image"].append(left_eye["img"])
            pose_angles = [right_eye["headpose_theta"], right_eye["headpose_phi"],
                           left_eye["headpose_theta"], left_eye["headpose_phi"]]
            data["pose"].append(pose_angles)
            data["landmarks"].append(landmarks)
            data["coordinates"].append(coords)
            counter += 1
            if counter % 100 == 0:
                print(counter)
            if counter == limit:
                return prepare_output(data)
    return prepare_output(data)


def create_mpiigaze_full_transformation_both_landmarks_coords(dataset_name, limit):
    """
    Creates dataset similiar to mpiigaze_processed_one_eye but tries to combine both eyes.
    """
    dataset_path = os.path.join(DATA_PATH, dataset_name)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    for person_id in range(15):
        print(f"person id: {person_id}")
        data = prepare_data_for_one_person(person_id, limit=limit)
        file_name = f"p{str(person_id).zfill(2)}.npz"
        file_path = os.path.join(dataset_path, file_name)
        with open(file_path, 'wb') as file:
            np.savez(file, right_image=data["right_image"], left_image=data["left_image"], pose=data["pose_landmarks"],
                     gaze=data["coordinates"])


if __name__ == '__main__':
    create_mpiigaze_full_transformation_both_landmarks_coords(dataset_name="mpiigaze_both_landmarks_coords", limit=3000)
