import os
import time

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_processing.mpii_face_gaze import extract_landmarks_from_annotation_file, load_mpii_face_gaze_image
from models.face_detectors.haarcascade_face_detector import HaarcascadeFaceDetector
from models.face_detectors.hog_face_detector import HogFaceDetector
from models.landmarks_detectors.kazemi_landmarks_detector import KazemiLandmarksDetector
from models.landmarks_detectors.landmarks_detector import filter_landmarks, MOUTH_EYES_CORNERS
from models.landmarks_detectors.lbf_landmarks_detector import LbfLandmarksDetector
from settings import RESULTS_PATH


def do_landmarks_experiment(face_detector, landmark_detector, verbose=False):
    meta_data = list()
    all_distances = list()

    # statistics
    results = dict()
    time_counter = 0
    face_false_positive = 0
    face_false_negative = 0

    for person_id in range(0, 15):
        print(f"Person id: {person_id}")

        # load true annotations
        file_names, true_landmarks = extract_landmarks_from_annotation_file(person_id)
        true_landmarks = np.asarray(true_landmarks)

        # collect data for evaluation
        indices = list()
        predicted_landmarks = list()

        for i, file_name in tqdm(enumerate(file_names)):
            # load image
            image = load_mpii_face_gaze_image(person_id, file_name)
            if image is not None:  # many annotations do not have their photos in dataset
                start_time = time.time()

                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                faces = face_detector.detect(image_gray)

                if len(faces) > 1:  # Take only first face
                    face_false_positive += (len(faces) - 1)
                    faces = faces[0].reshape(1, -1)
                elif len(faces) == 0:  # No face detected -> assume that face is on the full image
                    face_false_negative += 1
                    continue

                landmarks = landmark_detector.detect(image_gray, faces)

                elapsed = time.time() - start_time
                time_counter += elapsed

                # collect data
                landmarks = filter_landmarks(landmarks, MOUTH_EYES_CORNERS)
                predicted_landmarks.append(landmarks)
                indices.append(i)
                meta_data.append((person_id, file_name))

        # collect stats
        predicted_landmarks = np.asarray(predicted_landmarks).squeeze()
        true_landmarks = true_landmarks[indices, :-2, :]
        distances = np.linalg.norm(true_landmarks - predicted_landmarks, axis=2)
        all_distances.append(distances)
        results["p" + str(person_id).zfill(2) + "_mae"] = np.mean(distances)

        if verbose:
            print(meta_data)

    all_distances = np.concatenate(all_distances, axis=0)
    detected_faces_number = all_distances.shape[0] * all_distances.shape[1] / 6

    results["mae_macro"] = np.mean(list(results.values()))
    results["mae_micro"] = np.mean(all_distances)
    results["face_precision"] = detected_faces_number / (detected_faces_number + face_false_positive)
    results["face_recall"] = detected_faces_number / (detected_faces_number + face_false_negative)
    results["face_f1_score"] = 2 * results["face_precision"] * results["face_recall"] \
                               / (results["face_precision"] + results["face_recall"])
    results["time"] = time_counter / detected_faces_number
    return results


def do_experiments():
    face_detectors = [HaarcascadeFaceDetector(), HogFaceDetector()]
    landmarks_detectors = [LbfLandmarksDetector(), KazemiLandmarksDetector()]

    df_results = pd.DataFrame()
    results_dir = os.path.join(RESULTS_PATH, "face_landmarks")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "face_landmarks_detection_gpu.csv")

    for face_detector in face_detectors:
        for landmarks_detector in landmarks_detectors:
            row = {"face_detector": str(face_detector),
                   "landmarks_detector": str(landmarks_detector)}
            results = do_landmarks_experiment(face_detector, landmarks_detector)
            row.update(results)
            df_results = df_results.append([row], ignore_index=True)
            df_results.to_csv(results_path, index=False)


if __name__ == "__main__":
    do_experiments()
