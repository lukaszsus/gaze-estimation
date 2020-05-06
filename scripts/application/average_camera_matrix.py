import numpy as np
from scripts.create_dataset.create_dataset_mpiigaze_processed_both_rgb import load_camera_matrix

if __name__ == "__main__":
    matrices = list()
    for i in range(15):
        matrices.append(load_camera_matrix(path=f"Data/Original/p{str(i).zfill(2)}/Calibration/Camera.mat"))

    matrices = np.stack(matrices, axis=2)
    matrices = np.mean(matrices, axis=2)
    print(matrices)