import numpy as np
import cv2
from PIL import Image

from data_processing.head_pose import estimate_head_pose
from data_processing.algebraic_transformations import processed_both_eyes_rgb
from data_processing.utils import load_image_by_cv2, mpii_face_gaze_path_wrapper, mpiigaze_path_wrapper
from models.face_detectors.hog_face_detector import HogFaceDetector
from models.landmarks_detectors.kazemi_landmarks_detector import KazemiLandmarksDetector
from models.landmarks_detectors.landmarks_detector import filter_landmarks, MOUTH_EYES_CORNERS_HEAD_POSE
from scripts.create_dataset.create_dataset_mpiigaze_processed_both_rgb import load_face_model, load_camera_matrix, \
    load_screen_size


if __name__ == "__main__":
    face_model = load_face_model()
    eye_image_width = 60
    eye_image_height = 36

    person_id_str = "14"
    day = "day04"
    im_file = "0008.jpg"

    camera_matrix = load_camera_matrix(path=f"Data/Original/p{person_id_str}/Calibration/Camera.mat")
    screen_size = load_screen_size(path=f"Data/Original/p{person_id_str}/Calibration/screenSize.mat")
    im = load_image_by_cv2(mpii_face_gaze_path_wrapper(f"p{person_id_str}/{day}/{im_file}"))
    img = Image.fromarray(im, 'RGB')
    img.show()

    # convert image to Grayscale
    image_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    face_detector = HogFaceDetector()
    landmark_detector = KazemiLandmarksDetector()


    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = face_detector.detect(image_gray)
    if len(faces) > 1:      # Take only first face
        faces = faces[0].reshape(1, -1)

    # Detect landmarks on "image_gray"
    landmarks = landmark_detector.detect(image_gray, faces)
    landmarks = filter_landmarks(landmarks, indices=MOUTH_EYES_CORNERS_HEAD_POSE)

    headpose_hr, headpose_ht = estimate_head_pose(im, landmarks, camera_matrix, face_model_points=np.transpose(face_model))
    print(headpose_hr, headpose_ht)
    headpose_type = "2_3_dim_vectors"

    annotation = np.loadtxt(mpiigaze_path_wrapper(f"Data/Original/p{person_id_str}/{day}/annotation.txt"))
    true_headpose_hr = np.reshape(annotation[8 - 1, 29:32], (1, -1))
    true_headpose_ht = np.reshape(annotation[8 - 1, 32:35], (1, -1))
    print(true_headpose_hr, true_headpose_ht)

    right_eye_img, left_eye_img, headpose = processed_both_eyes_rgb(im, face_model, camera_matrix,
                                                                    headpose_hr, headpose_ht,
                                                                    headpose_type, eye_image_width=eye_image_width,
                                                                    eye_image_height=eye_image_height)

    img = Image.fromarray(right_eye_img, 'RGB')
    img.show()

    img = Image.fromarray(left_eye_img, 'RGB')
    img.show()
