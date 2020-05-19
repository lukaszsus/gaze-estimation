from functools import partial

from data_loader.hysts import load_hysts_mpiigaze_train_test_ds
from data_loader.mpiigaze_both_from_single import load_mpiigaze_train_test_ds_both_from_single, \
    load_mpiigaze_train_test_ds_both_leave_one_out, load_mpiigaze_train_test_ds_both_leave_one_out_reject_suspicious
from data_loader.mpiigaze_processed_loader import load_mpiigaze_train_test_2_angles_headpose
from data_loader.mpiigaze_processed_one_eye import load_mpiigaze_train_test_ds_one_eye, \
    load_mpiigaze_train_test_ds_one_eye_leave_one_out
from data_loader.own_dataset import load_own_dataset_one_person, load_train_test_ds_reject_suspicious

data_sets = {
    "hysts_mpii_gaze": {"name": "hysts_mpii_gaze",
                        "path": "hysts_mpiigazed_processed",
                        "dataset_size": "3000 per subject",
                        "input": ["eye_img", "headpose_2_angles"],
                        "eye_im_size": (36, 60),
                        "grayscale": True,
                        "headpose_size": 2,
                        "output": "angles",
                        "output_size": 2,
                        "load_function": partial(load_hysts_mpiigaze_train_test_ds,
                                                 dataset_name="hysts_mpiigazed_processed")},
    "mpiigaze_one_eye_grayscale": {"name": "mpiigaze_one_eye_grayscale",
                                   "path": "mpiigaze_processed_one_eye_like_hysts",
                                   "dataset_size": "3000 per subject",
                                   "input": ["eye_img", "headpose_2_angles"],
                                   "eye_im_size": (36, 60),
                                   "grayscale": True,
                                   "headpose_size": 2,
                                   "output": "angles",
                                   "output_size": 2,
                                   "load_function": partial(load_mpiigaze_train_test_ds_one_eye,
                                                            dataset_name="mpiigaze_processed_one_eye_like_hysts",
                                                            grayscale=True)},
    "mpiigaze_one_eye_rgb": {"name": "mpiigaze_one_eye_rgb",
                             "path": "mpiigaze_processed_one_eye_like_hysts",
                             "dataset_size": "3000 per subject",
                             "input": ["eye_img", "headpose_2_angles"],
                             "eye_im_size": (36, 60),
                             "grayscale": False,
                             "headpose_size": 2,
                             "output": "angles",
                             "output_size": 2,
                             "load_function": partial(load_mpiigaze_train_test_ds_one_eye,
                                                      dataset_name="mpiigaze_processed_one_eye_like_hysts",
                                                      grayscale=False)},
    "mpiigaze_both_from_single_grayscale": {"name": "mpiigaze_both_from_single_grayscale",
                                            "path": "mpiigaze_both_like_hysts",
                                            "dataset_size": "3000 per subject",
                                            "input": ["right_eye_img", "left_eye_img", "headpose_6_angles"],
                                            "eye_im_size": (36, 60),
                                            "grayscale": True,
                                            "headpose_size": 6,
                                            "output": "angles",
                                            "output_size": 2,
                                            "load_function": partial(load_mpiigaze_train_test_ds_both_from_single,
                                                                     dataset_name="mpiigaze_both_like_hysts",
                                                                     grayscale=True)},
    "mpiigaze_both_from_single_rgb": {"name": "mpiigaze_both_from_single_rgb",
                                      "path": "mpiigaze_both_like_hysts",
                                      "dataset_size": "3000 per subject",
                                      "input": ["right_eye_img", "left_eye_img", "headpose_6_angles"],
                                      "eye_im_size": (36, 60),
                                      "grayscale": False,
                                      "headpose_size": 6,
                                      "output": "angles",
                                      "output_size": 2,
                                      "load_function": partial(load_mpiigaze_train_test_ds_both_from_single,
                                                               dataset_name="mpiigaze_both_like_hysts",
                                                               grayscale=False)},
    "mpiigaze_processed_both_2anglesheadpose_grayscale": {"name": "mpiigaze_processed_both_2anglesheadpose_grayscale",
                                                          "path": "mpiigaze_processed_both_rgb_2_angles_headpose",
                                                          "dataset_size": None,
                                                          "input": ["right_eye_img", "left_eye_img",
                                                                    "headpose_2_angles"],
                                                          "eye_im_size": (36, 60),
                                                          "grayscale": True,
                                                          "headpose_size": 2,
                                                          "output": "angles",
                                                          "output_size": 2,
                                                          "load_function": partial(
                                                              load_mpiigaze_train_test_2_angles_headpose,
                                                              dataset_name="mpiigaze_processed_both_rgb_2_angles_headpose",
                                                              out_class="gaze",
                                                              grayscale=True)},
    "mpiigaze_processed_both_2anglesheadpose_rgb": {"name": "mpiigaze_processed_both_2anglesheadpose_rgb",
                                                    "path": "mpiigaze_processed_both_rgb_2_angles_headpose",
                                                    "dataset_size": None,
                                                    "input": ["right_eye_img", "left_eye_img",
                                                              "headpose_2_angles"],
                                                    "eye_im_size": (36, 60),
                                                    "grayscale": False,
                                                    "headpose_size": 2,
                                                    "output": "angles",
                                                    "output_size": 2,
                                                    "load_function": partial(
                                                        load_mpiigaze_train_test_2_angles_headpose,
                                                        dataset_name="mpiigaze_processed_both_rgb_2_angles_headpose",
                                                        out_class="gaze",
                                                        grayscale=False)},
    "mpiigaze_both_landmarks_coords_grayscale": {"name": "mpiigaze_both_landmarks_coords_grayscale",
                                                 "path": "mpiigaze_both_landmarks_coords",
                                                 "dataset_size": "3000 per subject",
                                                 "input": ["right_eye_img", "left_eye_img", "headpose_6_angles"],
                                                 "eye_im_size": (36, 60),
                                                 "grayscale": True,
                                                 "headpose_size": 30,
                                                 "output": "angles",
                                                 "output_size": 2,
                                                 "load_function": partial(load_mpiigaze_train_test_ds_both_from_single,
                                                                          dataset_name="mpiigaze_both_landmarks_coords",
                                                                          grayscale=True)},
    "mpiigaze_both_landmarks_coords_rgb": {"name": "mpiigaze_both_landmarks_coords_rgb",
                                           "path": "mpiigaze_both_landmarks_coords",
                                           "dataset_size": "3000 per subject",
                                           "input": ["right_eye_img", "left_eye_img", "headpose_6_angles"],
                                           "eye_im_size": (36, 60),
                                           "grayscale": False,
                                           "headpose_size": 30,
                                           "output": "angles",
                                           "output_size": 2,
                                           "load_function": partial(load_mpiigaze_train_test_ds_both_from_single,
                                                                    dataset_name="mpiigaze_both_landmarks_coords",
                                                                    grayscale=False)},
    "mpiigaze_one_eye_grayscale_leave_one_out": {"name": "mpiigaze_one_eye_grayscale_leave_one_out",
                                                 "path": "mpiigaze_processed",
                                                 "dataset_size": "3000 per subject",
                                                 "input": ["eye_img", "headpose_2_angles"],
                                                 "eye_im_size": (36, 60),
                                                 "grayscale": True,
                                                 "headpose_size": 2,
                                                 "output": "angles",
                                                 "output_size": 2,
                                                 "load_function": partial(
                                                     load_mpiigaze_train_test_ds_one_eye_leave_one_out,
                                                     dataset_name="mpiigaze_processed_one_eye_like_hysts",
                                                     grayscale=True)},
    "mpiigaze_both_grayscale_leave_one_out": {"name": "mpiigaze_both_from_single_grayscale",
                                              "path": "mpiigaze_both_like_hysts",
                                              "dataset_size": "3000 per subject",
                                              "input": ["right_eye_img", "left_eye_img", "headpose_6_angles"],
                                              "eye_im_size": (36, 60),
                                              "grayscale": True,
                                              "headpose_size": 6,
                                              "output": "angles",
                                              "output_size": 2,
                                              "load_function": partial(load_mpiigaze_train_test_ds_both_leave_one_out,
                                                                       dataset_name="mpiigaze_both_like_hysts",
                                                                       grayscale=True)},
    "mpiigaze_both_landmarks_coords_grayscale_leave_one_out": {"name": "mpiigaze_both_landmarks_coords_grayscale",
                                                               "path": "mpiigaze_both_landmarks_coords",
                                                               "dataset_size": "3000 per subject",
                                                               "input": ["right_eye_img", "left_eye_img",
                                                                         "headpose_6_angles"],
                                                               "eye_im_size": (36, 60),
                                                               "grayscale": True,
                                                               "headpose_size": 30,
                                                               "output": "angles",
                                                               "output_size": 2,
                                                               "load_function": partial(
                                                                   load_mpiigaze_train_test_ds_both_leave_one_out,
                                                                   dataset_name="mpiigaze_both_landmarks_coords",
                                                                   grayscale=True)},
    "mpiigaze_both_landmarks_coords_rgb_leave_one_out": {"name": "mpiigaze_both_landmarks_coords_rgb",
                                                         "path": "mpiigaze_both_landmarks_coords",
                                                         "dataset_size": "3000 per subject",
                                                         "input": ["right_eye_img", "left_eye_img",
                                                                   "headpose_6_angles"],
                                                         "eye_im_size": (36, 60),
                                                         "grayscale": False,
                                                         "headpose_size": 30,
                                                         "output": "angles",
                                                         "output_size": 2,
                                                         "load_function": partial(
                                                             load_mpiigaze_train_test_ds_both_leave_one_out,
                                                             dataset_name="mpiigaze_both_landmarks_coords",
                                                             grayscale=False)},
    "mpiigaze_both_landmarks_coords_grayscale_leave_one_out_reject_suspicious":
        {"name": "mpiigaze_both_landmarks_coords_grayscale",
         "path": "mpiigaze_both_landmarks_coords",
         "dataset_size": "3000 per subject",
         "input": ["right_eye_img", "left_eye_img",
                   "headpose_6_angles"],
         "eye_im_size": (36, 60),
         "grayscale": True,
         "headpose_size": 30,
         "output": "angles",
         "output_size": 2,
         "load_function": partial(
             load_mpiigaze_train_test_ds_both_leave_one_out_reject_suspicious,
             dataset_name="mpiigaze_both_landmarks_coords",
             grayscale=True)},
    "own_dataset_one_person": {"name": "own_dataset_one_person",
                               "path": "own_dataset_one_person",
                               "dataset_size": "5301 per subject",
                               "input": ["right_eye_img", "left_eye_img", "headpose_6_angles"],
                               "eye_im_size": (36, 60),
                               "grayscale": False,
                               "headpose_size": 30,
                               "output": "angles",
                               "output_size": 2,
                               "load_function": partial(load_own_dataset_one_person,
                                                        dataset_name="own_dataset",
                                                        grayscale=False)},
    "own_dataset_mpii_gaze": {"name": "own_dataset_mpii_gaze",
                              "path": "own_mpiigaze",
                              "dataset_size": "mpiigaze 3000 per subject",
                              "input": ["right_eye_img", "left_eye_img", "headpose_6_angles"],
                              "eye_im_size": (36, 60),
                              "grayscale": False,
                              "headpose_size": 30,
                              "output": "angles",
                              "output_size": 2,
                              "load_function": partial(load_train_test_ds_reject_suspicious,
                                                       dataset_name="own_mpiigaze",
                                                       grayscale=False)},
}
