import os
import numpy as np
from tqdm import tqdm
from application.pipeline import Pipeline
from application.utils import get_avg_camera_matrix
from data_processing.utils import mpii_face_gaze_path_wrapper, load_image_by_cv2
from models.face_detectors.hog_face_detector import HogFaceDetector
from models.landmarks_detectors.kazemi_landmarks_detector import KazemiLandmarksDetector
from models.model_loaders import load_best_modal3_conv_net
from scripts.create_dataset.create_dataset_mpiigaze_processed_both_rgb import load_camera_matrix, load_screen_size, \
    get_all_days, get_all_jpg_files, load_annotation
from settings import DATA_PATH


def get_all_days_ids(person_id):
    """
    Function retrieves days ids from strings like 'day{xx}'.
    :days_list_str: list containing days as str: day{xx} where xx is id
    """
    person_id_str = f"p{str(person_id).zfill(2)}"
    days_list_str = get_all_days(path=mpii_face_gaze_path_wrapper(f"{person_id_str}/"))
    ids = [int(day.replace("day", "")) for day in days_list_str]
    return ids


def get_all_images_ids_for_day(person_id, day_id):
    person_id_str = f"p{str(person_id).zfill(2)}"
    day = f"day{str(day_id).zfill(2)}"
    im_filenames = get_all_jpg_files(mpii_face_gaze_path_wrapper(f"{person_id_str}/{day}/"))
    ids = [int(filename.replace(".jpg", "")) for filename in im_filenames]
    return ids


def process_mpii_face_gaze():
    mean_camera_matrix = True

    right_images = list()
    left_images = list()
    poses = list()
    coords = list()
    people_ids = list()

    eye_image_width = 60
    eye_image_height = 36

    for person_id in range(15):
        person_id_str = str(person_id).zfill(2)
        print(f"Person id: {person_id_str}")

        if mean_camera_matrix:
            camera_matrix = get_avg_camera_matrix()
        else:
            camera_matrix = load_camera_matrix(path=f"Data/Original/p{person_id_str}/Calibration/Camera.mat")
        screen_size = load_screen_size(path=f"Data/Original/p{person_id_str}/Calibration/screenSize.mat")
        ann_path = mpii_face_gaze_path_wrapper(f"p{person_id_str}/p{person_id_str}.txt")
        annotation = np.loadtxt(ann_path, dtype=str)
        if len(annotation.shape) == 1:
            annotation = np.reshape(annotation, (1, -1))

        face_detector = HogFaceDetector()
        landmarks_detector = KazemiLandmarksDetector()
        model = load_best_modal3_conv_net()
        pipeline = Pipeline(gaze_estimation_model=model,
                            face_detector=face_detector,
                            landmarks_detector=landmarks_detector,
                            eye_image_width=eye_image_width,
                            eye_image_height=eye_image_height,
                            camera_matrix=camera_matrix,
                            screen_size=screen_size)

        file_paths = annotation[:, 0]

        for i, file_path in tqdm(enumerate(file_paths)):
            im = load_image_by_cv2(mpii_face_gaze_path_wrapper(f"p{person_id_str}/{file_path}"))
            data = pipeline.process(im)
            if data is None:
                continue
            right_images.append(data["right_image"])
            left_images.append(data["left_image"])
            poses.append(data["pose"])

            # coords
            coords_row = annotation[i, 1:3].astype(float)   # x, y
            coords_row = [coords_row[1], coords_row[0]]     # y, x
            coords_row = [coords_row[0] / screen_size[0], coords_row[1] / screen_size[1]]
            coords.append(coords_row)

            people_ids.append(person_id)

        save_data_to_file(right_images, left_images, poses, coords, people_ids)


def save_data_to_file(right_images_list, left_images_list, poses_list, coords_list, people_ids_list):
    right_images = np.concatenate(right_images_list)
    left_images = np.concatenate(left_images_list)
    poses = np.concatenate(poses_list)
    coords = np.stack(coords_list)

    dir_path = os.path.join(DATA_PATH, "mpii_face_gaze_mean_camera_matrix")
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "all_full.npz")
    np.savez(file_path, right_image=right_images, left_image=left_images, pose=poses,
             gaze=coords, person_id=people_ids_list)


if __name__ == "__main__":
    process_mpii_face_gaze()