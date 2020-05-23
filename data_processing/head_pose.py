import cv2
import numpy as np


def estimate_head_pose(im, landmarks, camera_matrix, face_model_points, show=False):
    if landmarks.dtype == np.int:
        landmarks = landmarks.astype(np.double)

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(face_model_points, landmarks, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    # change coordinate system orientation
    # translation_vector = np.multiply(translation_vector, [[-1], [-1], [-1]])

    if show:
        (line_start_point, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)
        (line_end_point, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 100.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)

        for p in landmarks:
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        # p1 = (int(landmarks[0][0]), int(landmarks[0][1]))
        p1 = (int(line_start_point[0][0][0]), int(line_start_point[0][0][1]))
        p2 = (int(line_end_point[0][0][0]), int(line_end_point[0][0][1]))

        cv2.line(im, p1, p2, (255, 0, 0), 2)

        # Display image
        cv2.imshow("Output", im)
        cv2.waitKey(0)

    return rotation_vector, translation_vector
