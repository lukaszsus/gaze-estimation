import os

import numpy as np
from PIL import Image

from data_processing.utils import load_image_by_cv2, mpii_face_gaze_path_wrapper
from settings import FOR_THESIS_DIR
from utils.landmarks import visualize_landmarks_mpii_gaze_format

if __name__ == "__main__":
    """
    Script prints all landmarks from MPII Gaze dataset on photo, to know which landmarks are they.
    """
    person_id_str = "14"
    day = "day04"
    im_file = "0008.jpg"
    offset = 740
    im = load_image_by_cv2(mpii_face_gaze_path_wrapper(f"p{person_id_str}/{day}/{im_file}"))

    annotation = np.loadtxt(mpii_face_gaze_path_wrapper(f"p{person_id_str}/p{person_id_str}.txt"), dtype=str)
    print(f"Real file path: {annotation[offset + 8 - 1, 0]}")
    landmarks = np.reshape(annotation[offset + 8 - 1, 3:15], (1, -1, 2))
    print(landmarks)
    landmarks = landmarks.astype(np.int)

    visualize_landmarks_mpii_gaze_format(landmarks, im, numbers=True)

    im = Image.fromarray(im)
    im.save(os.path.join(FOR_THESIS_DIR, "eye_landmarks.png"))
