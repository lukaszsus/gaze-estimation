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
