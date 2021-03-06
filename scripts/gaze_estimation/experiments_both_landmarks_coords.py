from utils.results_saver import save_parameters_to_comet, save_plots, save_table, save_metrics_to_comet, \
    save_weights_and_configuration
import os

from models.modal3_conv_net import Modal3ConvNet
from settings import RESULTS_PATH
from utils.datasets import data_sets
from datetime import datetime

from utils.metrics import final_predictions, final_test_measure_time
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

# USE_GPU = False  # just for logging some metrics correctly (for example forward_pass_time)
"""
That's for GPU training and maintaining one session and nice cuda lib loading.
"""
USE_GPU = True
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
AUTOTUNE = tf.data.experimental.AUTOTUNE


def perform_experiment_with_loaded_data(experiment_name,
                                        experiment_id,
                                        train_dataset, test_dataset,
                                        test_subject_ids,
                                        data_set,
                                        person_id,
                                        model_cls,
                                        epochs,
                                        conv_sizes,
                                        dense_sizes,
                                        dropout,
                                        optimizer_name,
                                        learning_rate,
                                        loss_name,
                                        track_angle_error,
                                        save_locally=True,
                                        start_datetime=None,
                                        use_gpu=True,
                                        save_model_path=None):
    if start_datetime is None:
        start_datetime = datetime.now()

    model = model_cls(conv_sizes=conv_sizes, dense_sizes=dense_sizes, dropout=dropout,
                      output_size=data_set["output_size"], track_angle_error=track_angle_error)
    experiment = save_parameters_to_comet(experiment_name, data_set=data_set, person_id=person_id,
                                          model_cls=model_cls,
                                          epochs=epochs,
                                          conv_sizes=conv_sizes,
                                          dense_sizes=dense_sizes,
                                          dropout=dropout,
                                          optimizer_name=optimizer_name,
                                          learning_rate=learning_rate,
                                          loss_name=loss_name)

    optimizer = tf.keras.optimizers.get(optimizer_name).from_config({"learning_rate": learning_rate})
    loss = tf.keras.losses.get(loss_name)
    model.compile(optimizer=optimizer,
                  loss=loss)
    history = model.fit(experiment=experiment,
                        train_dataset=train_dataset, test_dataset=test_dataset,
                        epochs=epochs,
                        start_datetime=start_datetime)

    # final metrics
    labels, predictions = final_predictions(model, test_dataset)
    forward_pass_time = final_test_measure_time(model=model, test_dataset=test_dataset)
    save_metrics_to_comet(experiment, labels=labels, predictions=predictions, test_subject_ids=test_subject_ids,
                          forward_pass_time=forward_pass_time, use_gpu=use_gpu, angle_error=False,
                          error_per_person=True)

    if save_locally:
        metrics = ["loss", "mean_absolute_error", "mean_squared_error", "mean_absolute_error_0",
                   "mean_absolute_error_1"]
        dir_path = os.path.join(RESULTS_PATH, data_set["path"])
        start_datetime_str = start_datetime.strftime("%Y-%m-%d-t%H-%M-%S")
        save_plots(dir_path, start_datetime_str, experiment_id, history, metrics)
        save_table(dir_path, start_datetime_str, experiment_id, experiment)
        # save_weights(dir_path, model=model, start_datetime_str=start_datetime_str, experiment_id=experiment_id)

    if save_model_path is not None:
        save_weights_and_configuration(save_model_path, model)

    del model
    experiment.end()


def main_experiments():
    # hyper parameters
    experiment_name = 'mpiigaze_both_landmarks_coords'
    track_angle_error = True
    experiment_id = 0
    start_datetime = datetime.now()
    # people_ids = list(range(15))
    people_ids = [None]
    experiments_data_set_names = ['mpiigaze_both_landmarks_coords_grayscale',
                                  'mpiigaze_both_landmarks_coords_rgb']
    val_split = 0.2
    models_cls = [Modal3ConvNet]
    num_epochs = [30]
    batch_sizes = [128]
    optimizers_names = ['Adam']
    learning_rates = [0.001, 0.0001]
    losses_names = ['mae']
    conv_sizes_list = [({"n_filters": 16, "filter_size": (5, 5), "padding": "valid", "stride": (1, 1),
                         "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)},
                        {"n_filters": 16, "filter_size": (5, 5), "padding": "valid", "stride": (1, 1),
                         "pool": None, "pool_size": None, "pool_stride": None},
                        {"n_filters": 16, "filter_size": (5, 5), "padding": "valid", "stride": (1, 1),
                         "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)}),

                       ({"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                         "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)},
                        {"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                         "pool": None, "pool_size": None, "pool_stride": None},
                        {"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                         "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)})
                       ]
    dense_sizes_list = [(256, 64, 2),
                        (512, 128, 2)]
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
                                                    use_gpu=USE_GPU
                                                )
                                                experiment_id = experiment_id + 1
                del train_dataset
                del test_dataset


def single_experiment_with_save_weights():
    save_model_path = None
    # save_model_path = "best_modal3_conv_net"

    # hyper parameters
    # experiment_name = 'mpiigaze_both_landmarks_coords'
    # experiment_name = 'own_dataset_only'
    experiment_name = 'own_all_dataset_mpii_gaze'
    track_angle_error = True
    experiment_id = 0
    start_datetime = datetime.now()
    people_ids = [None]
    # experiments_data_set_names = ['mpiigaze_both_landmarks_coords_rgb']
    experiments_data_set_names = ['own_dataset_mpii_gaze']
    val_split = 0.20
    models_cls = [Modal3ConvNet]
    num_epochs = [50]
    batch_sizes = [128]
    optimizers_names = ['Adam']
    learning_rates = [0.001]
    losses_names = ['mae']
    conv_sizes_list = [({"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                         "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)},
                        {"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                         "pool": None, "pool_size": None, "pool_stride": None},
                        {"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                         "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)})]
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


def experiments_leave_one_out():
    # hyper parameters
    experiment_name = 'mpiigaze_both_landmarks_coords_leave_one_out'
    track_angle_error = True
    experiment_id = 0
    start_datetime = datetime.now()
    people_ids = list(range(15))
    experiments_data_set_names = ['mpiigaze_both_landmarks_coords_grayscale_leave_one_out',
                                  'mpiigaze_both_landmarks_coords_rgb_leave_one_out']
    val_split = 0.2
    models_cls = [Modal3ConvNet]
    num_epochs = [30]
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
                                                    use_gpu=USE_GPU
                                                )
                                                experiment_id = experiment_id + 1
                del train_dataset
                del test_dataset


def experiments_leave_one_out_reject_suspicious():
    # hyper parameters
    experiment_name = 'mpiigaze_both_landmarks_coords_leave_one_out_reject_suspicious'
    track_angle_error = True
    experiment_id = 0
    start_datetime = datetime.now()
    people_ids = list(range(0, 7)) + list(range(8, 15))
    print(people_ids)
    experiments_data_set_names = ['mpiigaze_both_landmarks_coords_grayscale_leave_one_out_reject_suspicious',
                                  'mpiigaze_both_landmarks_coords_rgb_leave_one_out_reject_suspicious']
    val_split = 0.2
    models_cls = [Modal3ConvNet]
    num_epochs = [30]
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
                                                    use_gpu=USE_GPU
                                                )
                                                experiment_id = experiment_id + 1
                del train_dataset
                del test_dataset


def experiments_leave_one_out_full_reject_suspicious():
    save_model_path = "modal3_conv_net_full"
    # hyper parameters
    experiment_name = 'mpiigaze_both_landmarks_coords_full_reject_suspicious'
    track_angle_error = True
    experiment_id = 0
    start_datetime = datetime.now()
    people_ids = list(range(0, 7)) + list(range(8, 15))
    print(people_ids)
    experiments_data_set_names = ['mpiigaze_both_landmarks_coords_full_rgb_leave_one_out_reject_suspicious']
    val_split = 0.2
    models_cls = [Modal3ConvNet]
    num_epochs = [100]
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
                                                    save_model_path=save_model_path + f"_{person_id}"
                                                )
                                                experiment_id = experiment_id + 1
                del train_dataset
                del test_dataset


if __name__ == '__main__':
    # main_experiments()
    single_experiment_with_save_weights()
    # experiments_leave_one_out()
    # experiments_leave_one_out_reject_suspicious()
    # experiments_leave_one_out_full_reject_suspicious()