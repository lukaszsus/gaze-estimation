from unittest import TestCase
from data_processing.mpii_face_gaze import extract_landmarks_from_annotation_file, load_mpii_face_gaze_image
from utils.landmarks import visualize_landmarks, visualize_landmarks_mpii_gaze_format


# class TestMpiiFaceGaze(TestCase):
def test_load_mpii_face_gaze_image():
    hit_counter = 0
    counter = 0

    for person_id in range(15):
        file_names, landmarks = extract_landmarks_from_annotation_file(person_id)
        print(file_names)

        for file_name in file_names:
            im = load_mpii_face_gaze_image(person_id, file_name)
            if im is not None:
                hit_counter += 1
                print(f"Hit person: {person_id} file: {counter}")
            counter += 1

    print(hit_counter)
    print(counter)

    person_id = 0
    file_names, landmarks = extract_landmarks_from_annotation_file(person_id)
    im = load_mpii_face_gaze_image(0, file_names[19])
    visualize_landmarks_mpii_gaze_format([landmarks[19][:-2]], im)


if __name__ == "__main__":
    # test = TestMpiiFaceGaze()
    test_load_mpii_face_gaze_image()
