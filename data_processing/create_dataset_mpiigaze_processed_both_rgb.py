import os
import pickle

import cv2
import numpy as np
import scipy.io

from data_processing.mpiigaze_normalize_image import mpii_gaze_normalize_image
from data_processing.utils import load_image_by_cv2, mpiigaze_path_wrapper
from tqdm import tqdm

from settings import DATA_PATH


def load_face_model():
    path = mpiigaze_path_wrapper("6 points-based face model.mat")
    matdata = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    return matdata["model"]


def load_annotation(path):
    annotation = np.loadtxt(path)
    if len(annotation.shape) == 1:
        annotation = np.reshape(annotation, (1, -1))
    return annotation


def load_camera_matrix(path):
    path = mpiigaze_path_wrapper(path)
    matdata = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    return matdata["cameraMatrix"]


def load_screen_size(path, type="pixel"):
    """

    :param path:
    :param type: 'pixel' or 'mm' (millimeter)
    :return:
    """
    path = mpiigaze_path_wrapper(path)
    matdata = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    h, w = matdata[f"height_{type}"], matdata[f"width_{type}"]
    return (h, w)

def count_gaze_angles(gaze):
    gaze_theta = np.arcsin((-1) * gaze[1].squeeze())
    gaze_phi = np.arctan2((-1) * gaze[0].squeeze(), (-1) * gaze[2].squeeze())
    return gaze_theta, gaze_phi


def count_headpose_angles(headpose):
    M, _ = cv2.Rodrigues(headpose)
    Zv = M[:, 2]
    headpose_theta = np.arcsin(Zv[1])  # vertical head pose angle
    headpose_phi = np.arctan2(Zv[0], Zv[2])  # horizontal head pose angle
    return headpose_theta, headpose_phi


def get_img_gaze_headpose_per_eye(im, eye_center, head_rotation, gaze_target,
                                  eye_image_width, eye_image_height, camera_matrix):
    """
    :param im:
    :param eye_center:
    :param head_rotation:
    :param gaze_target:
    :param eye_image_width:
    :param eye_image_height:
    :param camera_matrix:
    :return: dictionary with img, gaze_theta, gaze_phi, headpose_theta, headpose_phi
    """
    eye_img, headpose, gaze = mpii_gaze_normalize_image(im, eye_center, head_rotation, gaze_target,
                                                        (eye_image_width, eye_image_height), camera_matrix)
    gaze_theta, gaze_phi = count_gaze_angles(gaze)
    headpose_theta, headpose_phi = count_headpose_angles(headpose)

    data = dict()
    data["img"] = eye_img
    data["gaze_theta"] = gaze_theta
    data["gaze_phi"] = gaze_phi
    data["headpose_theta"] = headpose_theta
    data["headpose_phi"] = headpose_phi

    return data


def parse_mpiigaze(person: int, day: str, img_n: str, eye_image_width=60, eye_image_height=36):
    face_model = load_face_model()

    person = str(person).zfill(2)
    day = str(day).zfill(2)
    img_n = str(img_n).zfill(4)
    im = load_image_by_cv2(mpiigaze_path_wrapper(f"Data/Original/p{person}/day{day}/{img_n}.jpg"))
    annotation = np.loadtxt(mpiigaze_path_wrapper(f"Data/Original/p{person}/day{day}/annotation.txt"))
    camera_matrix = load_camera_matrix(path=f"Data/Original/p{person}/Calibration/Camera.mat")

    headpose_hr = np.reshape(annotation[0, 29:32], (1, -1))
    headpose_ht = np.reshape(annotation[0, 32:35], (1, -1))
    h_r, _ = cv2.Rodrigues(headpose_hr)
    fc = np.dot(h_r, face_model)
    fc = fc + np.reshape(headpose_ht, (-1, 1))

    gaze_target = annotation[0, 26:29]
    gaze_target = np.reshape(gaze_target, (-1, 1))

    right_eye_center = 0.5 * (fc[:, 0] + fc[:, 1])
    left_eye_center = 0.5 * (fc[:, 2] + fc[:, 3])

    right_eye = get_img_gaze_headpose_per_eye(im, right_eye_center, h_r, gaze_target,
                                              eye_image_width, eye_image_height, camera_matrix)
    left_eye = get_img_gaze_headpose_per_eye(im, left_eye_center, h_r, gaze_target,
                                             eye_image_width, eye_image_height, camera_matrix)

    return right_eye, left_eye


def get_all_days(path):
    dirnames = os.listdir(path)
    day_dirs = [dirname for dirname in dirnames if "day" in dirname]
    return day_dirs


def get_all_jpg_files(path):
    filenames = os.listdir(path)
    filenames[:] = [filename for filename in filenames if "jpg" in filename]
    return filenames


# def save_coords(person_id, day, coordinates):
#     if not isinstance(coordinates, np.ndarray):
#         coordinates = np.array(coordinates)
#
#     data_list = [coordinates]
#     dirs = ["coordinates"]
#
#     dir_path = os.path.join(DATA_PATH, "mpiigaze_processed_both_rgb")
#     for i in range(len(data_list)):
#         data = data_list[i]
#         d = dirs[i]
#         path = os.path.join(dir_path, d, f"{d}_p{person_id}_{day}.pkl")
#         with open(path, "wb") as file:
#             pickle.dump(data, file)


def save_dataset_mpiigaze_processed_both_rgb(person_id, day, right_eyes, left_eyes, headposes, gazes, coordinates):
    right_eyes = np.array(right_eyes)
    left_eyes = np.array(left_eyes)
    headposes = np.array(headposes)
    gazes = np.array(gazes)
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)

    data_list = [right_eyes, left_eyes, headposes, gazes, coordinates]
    dirs = ["right_eye", "left_eye", "headpose", "gaze", "coordinates"]

    dir_path = os.path.join(DATA_PATH, "mpiigaze_processed_both_rgb")
    for i in range(len(data_list)):
        data = data_list[i]
        d = dirs[i]
        path = os.path.join(dir_path, d, f"{d}_p{person_id}_{day}.pkl")
        with open(path, "wb") as file:
            pickle.dump(data, file)


def create_dirs():
    dir_path = os.path.join(DATA_PATH, "mpiigaze_processed_both_rgb")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dirs = ["right_eye", "left_eye", "headpose", "gaze", "coordinates"]
    for d in dirs:
        path = os.path.join(dir_path, d)
        if not os.path.exists(path):
            os.makedirs(path)


def cut_eye(input_img, target_3d, h_r, roi_size, camera_matrix, focal_new=960, distance_new=600):
    distance = np.linalg.norm(target_3d)
    z_scale = distance_new / distance
    cam_new = np.array([[focal_new, 0, roi_size[0] / 2],
                        [0.0, focal_new, roi_size[1] / 2],
                        [0, 0, 1.0]])
    scale_mat = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, z_scale]])
    h_rx = h_r[:, 0]
    forward = target_3d / distance
    down = np.cross(forward, h_rx)
    down = down / np.linalg.norm(down)

    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)

    rot_mat = np.array([right, down, forward])
    warp_mat = np.dot((np.dot(cam_new, scale_mat)), (np.dot(rot_mat, np.linalg.inv(camera_matrix))))
    img_warped = cv2.warpPerspective(input_img, M=warp_mat, dsize=roi_size)

    return img_warped


def norm_coords(coords, screen_size):
    coords = np.array(coords)
    h = coords[:, 0]
    w = coords[:, 1]

    h = h / screen_size[0]
    w = w / screen_size[1]

    h = np.reshape(h, (-1, 1))
    w = np.reshape(w, (-1, 1))

    return np.concatenate((h, w), axis=1)


def create_dataset_mpiigaze_processed_both_rgb():
    """
    This function creates dataset with following record structure:
    (right_eye_rgb_img, left_eye_rgb_img, gaze_target_theta, gaze_target_phi, norm_x_coordinate, norm_y_coordinate).
    Records are saved to pickles. One pickle for one person_id_str.
    :return:
    """
    face_model = load_face_model()
    eye_image_width = 60
    eye_image_height = 36

    for person_id in range(15):
        person_id_str = str(person_id).zfill(2)
        print(f"--------------\nperson_id_str: {person_id_str}")
        camera_matrix = load_camera_matrix(path=f"Data/Original/p{person_id_str}/Calibration/Camera.mat")
        screen_size = load_screen_size(path=f"Data/Original/p{person_id_str}/Calibration/screenSize.mat")
        print(screen_size)
        day_dirs = get_all_days(path=mpiigaze_path_wrapper(f"Data/Original/p{person_id_str}/"))

        for day in day_dirs:
            left_eyes = list()
            right_eyes = list()
            headposes = list()
            gazes = list()
            coordinates = list()

            print(day)
            ann_path = mpiigaze_path_wrapper(f"Data/Original/p{person_id_str}/{day}/annotation.txt")
            annotation = load_annotation(ann_path)
            im_filenames = get_all_jpg_files(mpiigaze_path_wrapper(f"Data/Original/p{person_id_str}/{day}/"))
            for i in tqdm(range(len(im_filenames))):
                im_file = im_filenames[i]
                im = load_image_by_cv2(mpiigaze_path_wrapper(f"Data/Original/p{person_id_str}/{day}/{im_file}"))

                headpose_hr = np.reshape(annotation[i, 29:32], (1, -1))
                headpose_ht = np.reshape(annotation[i, 32:35], (1, -1))
                h_r, _ = cv2.Rodrigues(headpose_hr)
                fc = np.dot(h_r, face_model)
                fc = fc + np.reshape(headpose_ht, (-1, 1))

                gaze_target = annotation[i, 26:29]
                gaze_target = np.reshape(gaze_target, (-1, 1))

                right_eye_center = 0.5 * (fc[:, 0] + fc[:, 1])
                left_eye_center = 0.5 * (fc[:, 2] + fc[:, 3])

                right_eye_img = cut_eye(im, right_eye_center, h_r,
                                        (eye_image_width, eye_image_height), camera_matrix)
                left_eye_img = cut_eye(im, left_eye_center, h_r,
                                       (eye_image_width, eye_image_height), camera_matrix)

                right_eyes.append(right_eye_img)
                left_eyes.append(left_eye_img)
                headposes.append(np.concatenate((headpose_hr, headpose_ht), axis=1).squeeze())
                gazes.append(count_gaze_angles(gaze_target / np.linalg.norm(gaze_target)))
                coordinates.append([annotation[i, 25], annotation[i, 24]])

            coordinates = norm_coords(coordinates, screen_size)
            # save_coords(person_id_str, day, coordinates)
            save_dataset_mpiigaze_processed_both_rgb(person_id_str, day, right_eyes, left_eyes, headposes, gazes, coordinates)


if __name__ == '__main__':
    create_dirs()
    create_dataset_mpiigaze_processed_both_rgb()
