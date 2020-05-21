import os
import numpy as np
from tqdm import tqdm
from application.pipeline import Pipeline
from data_processing.utils import mpii_face_gaze_path_wrapper, load_image_by_cv2
from models.face_detectors.hog_face_detector import HogFaceDetector
from models.landmarks_detectors.kazemi_landmarks_detector import KazemiLandmarksDetector
from models.model_loaders import load_best_modal3_conv_net
from scripts.create_dataset.create_dataset_mpiigaze_processed_both_rgb import load_camera_matrix, load_screen_size, \
    get_all_days, get_all_jpg_files
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
    right_images = list()
    left_images = list()
    poses = list()

    eye_image_width = 60
    eye_image_height = 36

    for person_id in range(15):
        person_id_str = str(person_id).zfill(2)
        print(f"Person id: {person_id_str}")

        camera_matrix = load_camera_matrix(path=f"Data/Original/p{person_id_str}/Calibration/Camera.mat")
        screen_size = load_screen_size(path=f"Data/Original/p{person_id_str}/Calibration/screenSize.mat")
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

        days = get_all_days_ids(person_id)
        for day in tqdm(days):
            day_str = "day" + str(day).zfill(2)
            im_filenames = get_all_jpg_files(mpii_face_gaze_path_wrapper(f"p{person_id_str}/{day_str}/"))
            for im_file in im_filenames:
                im = load_image_by_cv2(mpii_face_gaze_path_wrapper(f"p{person_id_str}/{day_str}/{im_file}"))
                data = pipeline.process(im)
                if data is None:
                    continue
                right_images.append(data["right_image"])
                left_images.append(data["left_image"])
                poses.append(data["pose"])
        save_data_to_file(right_images, left_images, poses)


def save_data_to_file(right_images_list, left_images_list, poses_list):
    right_images = np.concatenate(right_images_list)
    left_images = np.concatenate(left_images_list)
    poses = np.concatenate(poses_list)

    dir_path = os.path.join(DATA_PATH, "mpii_face_gaze_processed")
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "all.npz")
    np.savez(file_path, right_image=right_images, left_image=left_images, pose=poses)


if __name__ == "__main__":
    process_mpii_face_gaze()