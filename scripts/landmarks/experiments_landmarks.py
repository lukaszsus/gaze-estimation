import os

import cv2
import numpy as np

from data_processing.mpii_face_gaze import extract_landmarks_from_annotation_file, load_mpii_face_gaze_image
from models.face_landmarks_detectors import get_haarcascade_detector, get_lbf_model, filter_lbf_model_landmarks
from settings import MPII_FACE_GAZE_PATH


def do_landmarks_experiments():
    detector = get_haarcascade_detector()
    landmark_detector = get_lbf_model()

    meta_data = list()
    all_distances = list()
    no_face_counter = 0
    for person_id in range(0, 15):
        # load true annotations
        file_names, true_landmarks = extract_landmarks_from_annotation_file(person_id)
        true_landmarks = np.asarray(true_landmarks)

        # collect data for evaluation
        indices = list()
        predicted_landmarks = list()
        for i, file_name in enumerate(file_names):
            # load image
            image = load_mpii_face_gaze_image(person_id, file_name)
            if image is not None:       # many annotations do not have their photos in dataset

                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                faces = detector.detectMultiScale(image_gray)

                if len(faces) > 1:  # Take only first face
                    faces = faces[0].reshape(1, -1)
                elif len(faces) == 0:   # No face detected -> assume that face is on the full image
                    no_face_counter += 1
                    continue

                _, landmarks = landmark_detector.fit(image_gray, faces)
                landmarks = filter_lbf_model_landmarks(landmarks)

                predicted_landmarks.append(landmarks)
                indices.append(i)
                meta_data.append((person_id, file_name))

        predicted_landmarks = np.asarray(predicted_landmarks)
        true_landmarks = true_landmarks[indices, :-2, :]
        distances = np.linalg.norm(true_landmarks - predicted_landmarks, axis=2)
        all_distances.append(distances)

    # find the biggest difference
    all_distances = np.concatenate(all_distances, axis=0)
    print(f"No face detected on {no_face_counter} images")
    print(f"Total number of images: {all_distances.shape[0] * all_distances.shape[1] / 6 + no_face_counter}")
    print(all_distances)
    max_dist_index = np.argmax(np.mean(all_distances, axis=1))
    print(meta_data[max_dist_index])


if __name__ == "__main__":
    do_landmarks_experiments()