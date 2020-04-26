from unittest import TestCase
from data_processing.mpii_face_gaze import extract_landmarks_from_metadata_file, load_mpii_face_gaze_image
from utils.landmarks import visualize_landmarks


# class TestMpiiFaceGaze(TestCase):
def test_load_mpii_face_gaze_image():
    person_id = 0
    file_names, landmarks = extract_landmarks_from_metadata_file(person_id)
    print(file_names)
    print(landmarks)

    im = load_mpii_face_gaze_image(person_id, file_names[0])
    visualize_landmarks(landmarks[0], im)


if __name__ == "__main__":
    # test = TestMpiiFaceGaze()
    test_load_mpii_face_gaze_image()
