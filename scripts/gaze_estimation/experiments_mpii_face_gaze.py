from datetime import datetime
from scripts.gaze_estimation.experiments_both_landmarks_coords import perform_experiment_with_loaded_data
from models.modal3_conv_net import Modal3ConvNet
from utils.datasets import data_sets

USE_GPU = True


def single_own_mpiigaze_save_weights():
    # save_model_path = 'modal3_conv_net_mean_camera_matrix'
    save_model_path = "modal3_conv_net_mpii_face_gaze"

    # hyper parameters
    experiment_name = 'own_mpii_face_gaze_dedicated_camera_matrix'      # 'mean_camera_matrix'
    track_angle_error = True
    experiment_id = 0
    start_datetime = datetime.now()
    people_ids = [None]     # people ids for leave one out
    experiments_data_set_names = ['mpii_face_gaze_own_pipeline']     # ['mean_camera_matrix']
    val_split = 0.2
    models_cls = [Modal3ConvNet]
    num_epochs = [75]
    batch_sizes = [128]
    optimizers_names = ['Adam']
    learning_rates = [0.001]
    losses_names = ['mae']
    conv_sizes_list = [({"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                         "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)},
                        {"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                         "pool": None, "pool_size": None, "pool_stride": None},
                        {"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                         "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)})
                       ]
    dense_sizes_list = [(512, 128, 2)]
    dropouts = [0.1]

    for person_id in people_ids:
        for batch_size in batch_sizes:
            for data_set_name in experiments_data_set_names:
                data_set = data_sets[data_set_name]
                data_set_loader = data_set["load_function"]
                train_dataset, test_dataset, test_subject_ids = data_set_loader(person_id=person_id,
                                                                                val_split=val_split,
                                                                                batch_size=batch_size)

                for model_cls in models_cls:
                    for epochs in num_epochs:
                        for conv_sizes in conv_sizes_list:
                            for dense_sizes in dense_sizes_list:
                                for dropout in dropouts:
                                    for optimizer_name in optimizers_names:
                                        for learning_rate in learning_rates:
                                            for loss_name in losses_names:
                                                print(f"Experiment id: {experiment_id}")
                                                perform_experiment_with_loaded_data(
                                                    experiment_name,
                                                    experiment_id,
                                                    train_dataset, test_dataset,
                                                    test_subject_ids=test_subject_ids,
                                                    person_id=person_id,
                                                    start_datetime=start_datetime,
                                                    data_set=data_set,
                                                    model_cls=model_cls,
                                                    epochs=epochs,
                                                    conv_sizes=conv_sizes,
                                                    dense_sizes=dense_sizes,
                                                    dropout=dropout,
                                                    optimizer_name=optimizer_name,
                                                    learning_rate=learning_rate,
                                                    loss_name=loss_name,
                                                    track_angle_error=track_angle_error,
                                                    use_gpu=USE_GPU,
                                                    save_model_path=save_model_path
                                                )
                                                experiment_id = experiment_id + 1
                del train_dataset
                del test_dataset


if __name__ == "__main__":
    single_own_mpiigaze_save_weights()