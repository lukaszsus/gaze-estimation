import cv2

from data_processing.mpii_face_gaze import extract_landmarks_from_annotation_file, load_mpii_face_gaze_image
from models.face_landmarks_detectors import get_haarcascade_detector, get_lbf_model, filter_lbf_model_landmarks
from utils.landmarks import visualize_landmarks

if __name__ == "__main__":
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
    predicted_landmarks = filter_lbf_model_landmarks(predicted_landmarks)
    image_landmarks = image.copy()
    visualize_landmarks([true_landmarks], image_landmarks)
    visualize_landmarks([predicted_landmarks], image_landmarks, color=(255, 0, 0))