import numpy as np

from application.pipeline import Pipeline
from data_processing.utils import mpii_face_gaze_path_wrapper, load_image_by_cv2, mpiigaze_path_wrapper
from models.face_detectors.hog_face_detector import HogFaceDetector
from models.landmarks_detectors.kazemi_landmarks_detector import KazemiLandmarksDetector
from scripts.create_dataset.create_dataset_mpiigaze_processed_both_rgb import load_camera_matrix, \
    load_screen_size

if __name__ == "__main__":
    eye_image_width = 60
    eye_image_height = 36

    # person_id_str = "00"
    # day = "day01"
    # im_file = "0484.jpg"

    person_id_str = "14"
    day = "day04"
    im_file = "0008.jpg"

    camera_matrix = load_camera_matrix(path=f"Data/Original/p{person_id_str}/Calibration/Camera.mat")
    screen_size = load_screen_size(path=f"Data/Original/p{person_id_str}/Calibration/screenSize.mat")
    annotation = np.loadtxt(mpiigaze_path_wrapper(f"Data/Original/p{person_id_str}/{day}/annotation.txt"))
    im = load_image_by_cv2(mpii_face_gaze_path_wrapper(f"p{person_id_str}/{day}/{im_file}"))

    face_detector = HogFaceDetector()
    landmarks_detector = KazemiLandmarksDetector()

    pipeline = Pipeline(face_detector=face_detector,
                        landmarks_detector=landmarks_detector,
                        eye_image_width=eye_image_width,
                        eye_image_height=eye_image_height,
                        camera_matrix=camera_matrix,
                        screen_size=screen_size)

    prediction = pipeline.predict(im)

    print(prediction)

    print(f"Coorect prediction (x, y): {annotation[8 - 1, 25]}, {annotation[8 - 1, 24]}")