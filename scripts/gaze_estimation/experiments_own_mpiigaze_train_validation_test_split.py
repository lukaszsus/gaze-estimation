from data_loader.validation_with_loaded_model import perform_test_with_loaded_data
from models.model_loaders import load_best_modal3_conv_net
from scripts.gaze_estimation.experiments_both_landmarks_coords import perform_experiment_with_loaded_data, USE_GPU
from datetime import datetime
from models.modal3_conv_net import Modal3ConvNet
from utils.datasets import data_sets


def single_own_mpiigaze_full_save_weights(hiper_param_epochs):
    save_model_path = f"modal3_conv_net_own_mpiigaze_train_val_test_{hiper_param_epochs}_epochs"

    # hyper parameters
    experiment_name = 'own_mpii_full_train_val_test_split'
    track_angle_error = True
    experiment_id = 0
    start_datetime = datetime.now()
    people_ids = [None]     # people ids for leave one out
    experiments_data_set_names = ['own_mpiigaze_full_train_val']
    val_split = 0.2
    models_cls = [Modal3ConvNet]
    num_epochs = [hiper_param_epochs]
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


def check_error_on_test_data(hiper_param_epochs):
    # hyper parameters
    experiment_name = 'own_mpii_full_train_val_test_split'
    experiment_id = 0
    experiments_data_set_names = ['own_mpiigaze_full_train_test']
    batch_size = 128

    model = load_best_modal3_conv_net(test=False,
                                      file_name=f"modal3_conv_net_own_mpiigaze_train_val_test_{hiper_param_epochs}_epochs.h5")

    for data_set_name in experiments_data_set_names:
        data_set = data_sets[data_set_name]
        data_set_loader = data_set["load_function"]
        test_dataset, test_subject_ids = data_set_loader(person_id=None,
                                                         val_split=None,
                                                         batch_size=batch_size)
        print(f"Experiment id: {experiment_id}")
        perform_test_with_loaded_data(
            experiment_name=experiment_name,
            test_dataset=test_dataset,
            test_subject_ids=test_subject_ids,
            data_set=data_set,
            model=model,
            model_cls=Modal3ConvNet
        )
        experiment_id = experiment_id + 1
        del test_dataset


if __name__ == "__main__":
    for epochs in [200, 500]:
        single_own_mpiigaze_full_save_weights(epochs)
        check_error_on_test_data(epochs)