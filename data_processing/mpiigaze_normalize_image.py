import cv2
import numpy as np


def mpii_gaze_normalize_image(input_img, target_3d, h_r, gc, roi_size, camera_matrix, focal_new=960, distance_new=600):
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

    # rotation normalization
    cnv_mat = np.dot(scale_mat, rot_mat)
    h_r_new = np.dot(cnv_mat, h_r)
    hrnew, _ = cv2.Rodrigues(h_r_new)
    htnew = np.dot(cnv_mat, target_3d)
    htnew = np.reshape(htnew, (-1, 1))

    # gaze vector_normalization
    gcnew = np.dot(cnv_mat, gc)
    gvnew = gcnew - htnew
    gvnew = gvnew / np.linalg.norm(gvnew)

    return img_warped, hrnew, gvnew


def mpii_gaze_normalize_image_without_gaze(input_img, target_3d, h_r, roi_size, camera_matrix, focal_new=960, distance_new=600):
    """
    :param input_img: np.ndarray(720, 1280, 3) example shape - image with full face (full image from camera)
    :param target_3d: np.ndarray(3, ) - 3d eye center localization; localization is estimated with 3d face model
    :param h_r: np.ndarray(3, 3) - head rotation array - it is estimated head rotation matrix multiplied (dot operation)
                with face model
    :param roi_size: tuple(eye_image_width, eye_image_height)
    :param camera_matrix: np.ndarray(3, 3) - camera matrix tells about camera distortions and position
    :param focal_new: param for new camera matrix
    :param distance_new: param for new camera matrix
    """
    distance = np.linalg.norm(target_3d)        # distance from camera to eye center
    z_scale = distance_new / distance   # we want to simulate that the eye is 600 mm from camera
    cam_new = np.array([[focal_new, 0, roi_size[0] / 2],
                        [0.0, focal_new, roi_size[1] / 2],
                        [0, 0, 1.0]])       # standard camera matrix for our transformations
    scale_mat = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, z_scale]])     # scaling of image
    h_rx = h_r[:, 0]
    forward = target_3d / distance      # truly it is only direction of eye center
    down = np.cross(forward, h_rx)      # perpendicular vector to forward
    down = down / np.linalg.norm(down)      # normalization

    right = np.cross(down, forward)     # perpendicular vector to forward and down
    right = right / np.linalg.norm(right)   # normalization

    rot_mat = np.array([right, down, forward])
    # (np.dot(cam_new, scale_mat)) is (3, 3) matrix
    # (np.dot(rot_mat, np.linalg.inv(camera_matrix))) is (3, 3) matrix
    # => warp_mat is also (3, 3) matrix
    warp_mat = np.dot((np.dot(cam_new, scale_mat)), (np.dot(rot_mat, np.linalg.inv(camera_matrix))))
    # image transformed to be vertical and horizontal
    img_warped = cv2.warpPerspective(input_img, M=warp_mat, dsize=roi_size)

    # rotation normalization
    cnv_mat = np.dot(scale_mat, rot_mat)    # eye rotation matrix scaled
    h_r_new = np.dot(cnv_mat, h_r)
    hrnew, _ = cv2.Rodrigues(h_r_new)

    return img_warped, hrnew
