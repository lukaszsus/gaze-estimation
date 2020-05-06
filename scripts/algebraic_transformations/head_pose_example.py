import cv2
import numpy as np
from PIL import Image

from data_processing.head_pose import estimate_head_pose
from data_processing.utils import load_image_by_cv2, mpii_face_gaze_path_wrapper
from models.face_detectors.hog_face_detector import HogFaceDetector
from models.landmarks_detectors.kazemi_landmarks_detector import KazemiLandmarksDetector
from models.landmarks_detectors.landmarks_detector import filter_landmarks, MOUTH_EYES_CORNERS, \
    MOUTH_EYES_CORNERS_HEAD_POSE, HEAD_POSE
from scripts.create_dataset.create_dataset_mpiigaze_processed_both_rgb import load_camera_matrix, load_screen_size, \
    load_face_model

if __name__ == "__main__":
    face_model = load_face_model()

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
    landmarks = filter_landmarks(landmarks, indices=MOUTH_EYES_CORNERS)

    print(estimate_head_pose(im, landmarks, camera_matrix, face_model_points=np.transpose(face_model), show=True))