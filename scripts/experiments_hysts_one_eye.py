from utils.results_saver import save_parameters_to_comet, save_results_locally

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from models.modal2_conv_net import Modal2ConvNet
from utils.datasets import data_sets
from datetime import datetime

"""
That's for GPU training and maintaining one session and nice cuda lib loading.
"""
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
AUTOTUNE = tf.data.experimental.AUTOTUNE


def perform_experiment_with_loaded_data(experiment_name,
                                        experiment_id,
                                        train_dataset, test_dataset,
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
                                        save_locally=True,
                                        start_time=None):
    if start_time is None:
        start_time = datetime.now()

    model = model_cls(conv_sizes=conv_sizes, dense_sizes=dense_sizes, dropout=dropout,
                      output_size=data_set["output_size"])
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
    history = model.fit(
        experiment=experiment,
        train_dataset=train_dataset, test_dataset=test_dataset,
        epochs=epochs,
        start_time=start_time)

    if save_locally:
        metrics = ["loss", "mean_absolute_error", "mean_squared_error", "mean_absolute_error_0",
                   "mean_absolute_error_1"]
        save_results_locally(
            name=data_set["path"], experiment=experiment, start_time=start_time, experiment_id=experiment_id,
            model=model, metrics=metrics, history=history)

    experiment.end()


def main_experiments():
    data_size = None  # None means 'take all data'

    # hyper parameters
    experiment_name = "hysts_one_eye"
    experiment_id = 0
    start_time = datetime.now()
    people_ids = list(range(15))
    experiments_data_set_names = ['hysts_mpii_gaze']
    val_split = 0.2
    models_cls = [Modal2ConvNet]
    num_epochs = [20]
    batch_sizes = [128]
    optimizers_names = ['Adam', 'SGD']
    learning_rates = [0.01, 0.003, 0.001, 0.0001]
    losses_names = ['mse', 'mae']
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
                         "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)}),

                       ({"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                         "pool": None, "pool_size": None, "pool_stride": None},
                        {"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                         "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)},
                        {"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                         "pool": None, "pool_size": None, "pool_stride": None},
                        {"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                         "pool": None, "pool_size": None, "pool_stride": None},
                        {"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                         "pool": None, "pool_size": None, "pool_stride": None},
                        {"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                         "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)})
                       ]
    dense_sizes_list = [(256, 64, 2),
                        (256, 128, 64, 2),
                        (512, 128, 2)]
    dropouts = [0.1]

    for person_id in people_ids:
        for batch_size in batch_sizes:
            for data_set_name in experiments_data_set_names:
                data_set = data_sets[data_set_name]
                data_set_loader = data_set["load_function"]
                train_dataset, test_dataset = data_set_loader(person=person_id, val_split=val_split,
                                                              batch_size=batch_size)

                for model_cls in models_cls:
                    for epochs in num_epochs:
                        for conv_sizes in conv_sizes_list:
                            for dense_sizes in dense_sizes_list:
                                for dropout in dropouts:
                                    for optimizer_name in optimizers_names:
                                        for learning_rate in learning_rates:
                                            for loss_name in losses_names:
                                                perform_experiment_with_loaded_data(
                                                    experiment_name,
                                                    experiment_id,
                                                    train_dataset, test_dataset,
                                                    person_id=person_id,
                                                    start_time=start_time,
                                                    data_set=data_set,
                                                    model_cls=model_cls,
                                                    epochs=epochs,
                                                    conv_sizes=conv_sizes,
                                                    dense_sizes=dense_sizes,
                                                    dropout=dropout,
                                                    optimizer_name=optimizer_name,
                                                    learning_rate=learning_rate,
                                                    loss_name=loss_name
                                                )
                                                experiment_id = experiment_id + 1


if __name__ == '__main__':
    main_experiments()
