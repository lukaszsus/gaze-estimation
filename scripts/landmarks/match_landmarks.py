import cv2
from data_processing.mpii_face_gaze import extract_landmarks_from_annotation_file, load_mpii_face_gaze_image
from models.face_detectors.hog_face_detector import HogFaceDetector
from models.face_landmarks_detectors import get_haarcascade_detector, get_lbf_model
from models.landmarks_detectors.kazemi_landmarks_detector import KazemiLandmarksDetector
from models.landmarks_detectors.lbf_landmarks_detector import filter_lbf_model_landmarks
from utils.landmarks import visualize_faces, visualize_landmarks


def haarcascade_lbf():
    person_id = 0
    file_names, landmarks = extract_landmarks_from_annotation_file(person_id)
    image = load_mpii_face_gaze_image(0, file_names[417])

    # convert image to Grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = get_haarcascade_detector()
    landmark_detector = get_lbf_model()

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(image_gray)
    if len(faces) > 1:      # Take only first face
        faces = faces[0].reshape(1, -1)
    visualize_faces(faces, image.copy())

    # Detect landmarks on "image_gray"
    _, landmarks = landmark_detector.fit(image_gray, faces)
    landmarks = filter_lbf_model_landmarks(landmarks)
    visualize_landmarks([landmarks], image.copy())


def hog_kazemi():
    person_id = 0
    file_names, landmarks = extract_landmarks_from_annotation_file(person_id)
    image = load_mpii_face_gaze_image(0, file_names[417])

    # convert image to Grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_detector = HogFaceDetector()
    landmark_detector = KazemiLandmarksDetector()

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = face_detector.detect(image_gray)
    if len(faces) > 1:      # Take only first face
        faces = faces[0].reshape(1, -1)
    visualize_faces(faces, image.copy())

    # Detect landmarks on "image_gray"
    landmarks = landmark_detector.detect(image_gray, faces)
    visualize_landmarks(landmarks, image.copy())


if __name__ == "__main__":
    # haarcascade_lbf()
    hog_kazemi()
