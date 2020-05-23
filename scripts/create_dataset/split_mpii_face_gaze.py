import os
import numpy as np

from settings import DATA_PATH


def split_mpii_face_gaze():
    # dir_path = os.path.join(DATA_PATH, "mpii_face_gaze_processed")
    dir_path = os.path.join(DATA_PATH, "mpii_face_gaze_mean_camera_matrix")
    file_path = os.path.join(dir_path, "all_full.npz")
    data = np.load(file_path)
    right_images = data["right_image"]
    left_images = data["left_image"]
    poses = data["pose"]
    gazes = data["gaze"]
    test_subject_ids = data["person_id"]

    people_ids = np.unique(test_subject_ids)

    for person_id in people_ids:
        mask = test_subject_ids == person_id
        out_path = os.path.join(dir_path, "p" + str(person_id).zfill(2) + ".npz")
        np.savez(out_path, right_image=right_images[mask],
                 left_image=left_images[mask], pose=poses[mask],
                 gaze=gazes[mask])


if __name__ == "__main__":
    split_mpii_face_gaze()