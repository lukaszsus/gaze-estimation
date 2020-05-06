import cv2
import numpy as np
from data_processing.mpii_face_gaze import extract_landmarks_from_annotation_file, load_mpii_face_gaze_image
from models.face_detectors.haarcascade_face_detector import HaarcascadeFaceDetector
from models.face_detectors.hog_face_detector import HogFaceDetector
from models.face_landmarks_detectors import get_haarcascade_detector, get_lbf_model
from models.landmarks_detectors.lbf_landmarks_detector import LbfLandmarksDetector
from models.landmarks_detectors.landmarks_detector import filter_landmarks, MOUTH_EYES_CORNERS
from utils.landmarks import visualize_landmarks, visualize_faces


def haarcascade_lbf_example():
    person_id = 13
    file_name = "day05/0032.jpg"
    file_names, true_landmarks = extract_landmarks_from_annotation_file(person_id)
    image = load_mpii_face_gaze_image(person_id, file_name)

    # convert image to Grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    detector = get_haarcascade_detector()
    landmark_detector = get_lbf_model()

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(image_gray)
    if len(faces) > 1:  # Take only first face
        faces = faces[0].reshape(1, -1)

    # Detect landmarks on "image_gray"
    _, predicted_landmarks = landmark_detector.fit(image_gray, faces)
    predicted_landmarks = filter_landmarks(predicted_landmarks, indices=MOUTH_EYES_CORNERS)
    image_landmarks = image.copy()
    visualize_landmarks([true_landmarks], image_landmarks)
    visualize_landmarks([predicted_landmarks], image_landmarks, color=(255, 0, 0))


def hog_lbf_example():
    person_id = 0
    file_name = "day39/0108.jpg"
    file_names, true_landmarks = extract_landmarks_from_annotation_file(person_id)
    image = load_mpii_face_gaze_image(person_id, file_name)
    true_landmarks = np.asarray(true_landmarks)
    true_landmarks = true_landmarks[34, :-2, :]

    # convert image to Grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    face_detector = HogFaceDetector()
    landmark_detector = LbfLandmarksDetector()

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = face_detector.detect(image_gray)
    if len(faces) > 1:  # Take only first face
        faces = faces[0].reshape(1, -1)
    visualize_faces(faces, image)

    # Detect landmarks on "image_gray"
    predicted_landmarks = landmark_detector.detect(image_gray, faces)
    image_landmarks = image.copy()
    visualize_landmarks([true_landmarks], image_landmarks)
    visualize_landmarks([predicted_landmarks], image_landmarks, color=(255, 0, 0))

if __name__ == "__main__":
    # haarcascade_lbf_example()
    hog_lbf_example()