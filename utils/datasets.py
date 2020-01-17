from data_loader.hysts import load_hysts_mpiigaze_train_test_ds_generator

data_sets = {
    "hysts_mpii_gaze": {"path": "hysts_mpiigazed_processed",
                        "input": ["eye_img", "headpose_2_angles"],
                        "eye_im_size": (36, 60),
                        "grayscale": True,
                        "headpose_size": 2,
                        "eye_combination_type": None,      # separately or stacked
                        "output": "angles",
                        "output_size": 2,
                        "load_function": load_hysts_mpiigaze_train_test_ds_generator},
}
