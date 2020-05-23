from scripts.create_dataset.create_dataset_mpiigaze_processed_both_rgb import load_screen_size


def check_resolution():
    for person_id in range(0, 15):
        print(f"person id: {person_id}")
        person_id_str = str(person_id).zfill(2)
        screen_size = load_screen_size(path=f"Data/Original/p{person_id_str}/Calibration/screenSize.mat")
        print(screen_size)


if __name__ == "__main__":
    check_resolution()