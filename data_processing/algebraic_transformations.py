import cv2
import numpy as np

from data_processing.mpiigaze_normalize_image import mpii_gaze_normalize_image, mpii_gaze_normalize_image_without_gaze
from scripts.create_dataset.create_dataset_mpiigaze_processed_both_rgb import cut_eye, count_headpose_angles, \
    norm_landmarks


def processed_both_eyes_rgb(im, face_model, camera_matrix, headpose_hr, headpose_ht, headpose_type, eye_image_width=60,
                            eye_image_height=36):
    h_r, _ = cv2.Rodrigues(headpose_hr)
    fc = np.dot(h_r, face_model)
    fc = fc + np.reshape(headpose_ht, (-1, 1))

    right_eye_center = 0.5 * (fc[:, 0] + fc[:, 1])
    left_eye_center = 0.5 * (fc[:, 2] + fc[:, 3])

    right_eye_img = cut_eye(im, right_eye_center, h_r,
                            (eye_image_width, eye_image_height), camera_matrix)
    left_eye_img = cut_eye(im, left_eye_center, h_r,
                           (eye_image_width, eye_image_height), camera_matrix)

    if headpose_type == "2_3_dim_vectors":
        headpose = np.concatenate((headpose_hr, headpose_ht), axis=1).squeeze()
    elif headpose_type == "2_angles":
        headpose = count_headpose_angles(headpose_hr / np.linalg.norm(headpose_hr))

    return right_eye_img, left_eye_img, headpose


def parse_both_eyes_rgb_landmark(im, face_model, camera_matrix, headpose_hr, headpose_ht,
                                 eye_image_width=60, eye_image_height=36):
    """
    Very similar to parse_mpiigaze_landmark_coords but without gaze (answers in general) and landmarks.
    """
    im_height, im_width, _ = im.shape
    h_r, _ = cv2.Rodrigues(headpose_hr)
    fc = np.dot(h_r, face_model)
    fc = fc + np.reshape(headpose_ht, (-1, 1))

    right_eye_center = 0.5 * (fc[:, 0] + fc[:, 1])
    left_eye_center = 0.5 * (fc[:, 2] + fc[:, 3])

    right_eye_img, right_eye_theta, right_eye_phi = _get_img_gaze_headpose_per_eye(im, right_eye_center, h_r,
                                                                                   eye_image_width, eye_image_height,
                                                                                   camera_matrix)
    left_eye_img, left_eye_theta, left_eye_phi = _get_img_gaze_headpose_per_eye(im, left_eye_center, h_r,
                                                                                eye_image_width, eye_image_height,
                                                                                camera_matrix)

    pose_angles = [right_eye_theta, right_eye_phi,
                   left_eye_theta, left_eye_phi]
    data = {"right_image": right_eye_img,
            "left_image": left_eye_img,
            "pose": pose_angles}

    return data


def _get_img_gaze_headpose_per_eye(im, eye_center, head_rotation, eye_image_width, eye_image_height, camera_matrix):
    eye_img, headpose = mpii_gaze_normalize_image_without_gaze(im, eye_center, head_rotation,
                                                               (eye_image_width, eye_image_height), camera_matrix)
    headpose_theta, headpose_phi = count_headpose_angles(headpose)

    return eye_img, headpose_theta, headpose_phi
