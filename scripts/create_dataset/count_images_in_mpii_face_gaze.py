from data_processing.utils import mpii_face_gaze_path_wrapper
from scripts.create_dataset.create_dataset_mpiigaze_processed_both_rgb import get_all_days, get_all_jpg_files


if __name__ == "__main__":
    counter = 0

    for person_id in range(15):
        print(f"person id: {person_id}")
        person_id_str = f"p{str(person_id).zfill(2)}"
        days_list_str = get_all_days(path=mpii_face_gaze_path_wrapper(f"{person_id_str}/"))
        # days_ids = [int(day.replace("day", "")) for day in days_list_str]
        for day in days_list_str:
            im_filenames = get_all_jpg_files(mpii_face_gaze_path_wrapper(f"{person_id_str}/{day}/"))
            im_ids = [int(filename.replace(".jpg", "")) for filename in im_filenames]
            for im_id in im_ids:
                counter += 1

    print(counter)