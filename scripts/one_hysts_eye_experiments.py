#!/usr/bin/python3
import os
import time

import pandas as pd
import tensorflow as tf
from datetime import datetime
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

"""
That's for GPU training and maintaining one session and nice cuda lib loading.
"""
config = ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.75
session = InteractiveSession(config=config)

from data_loader.hysts import load_hysts_mpiigaze_train_test_ds_generator
from models.modal2_conv_net import Modal2ConvNet
from utils.plots import plot_metrics, save_summary, create_dirs, RESULTS_PATH


def do_experiments():
    input_size = {"height": 36, "width": 60, "num_channels": 3}
    models_params = list()

    models_cls = [Modal2ConvNet]
    out_classes = ["gaze"]
    grayscales = [True]
    num_epochs = [20]
    batch_sizes = [128]
    optimizers_cls = ['Adam', 'SGD']
    learning_rates = [0.01, 0.003, 0.001, 0.0001]
    losses = ['mse', 'mae']
    conv_sizes_list = [({"n_filters": 16, "filter_size": (5, 5), "padding": "valid", "stride": (1, 1),
                         "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)},
                        {"n_filters": 16, "filter_size": (5, 5), "padding": "valid", "stride": (1, 1),
                         "pool": None, "pool_size": None, "pool_stride": None},
                        {"n_filters": 16, "filter_size": (5, 5), "padding": "valid", "stride": (1, 1),
                         "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)})]
    dense_sizes_list = [(256, 64, 2)]
    dropouts = [0.1]

    i = 0
    for model_cls in models_cls:
        for out_class in out_classes:
            for grayscale in grayscales:
                for conv_sizes in conv_sizes_list:
                    for dense_sizes in dense_sizes_list:
                        for dropout in dropouts:
                            for epochs in num_epochs:
                                for batch_size in batch_sizes:
                                    for optimizer_cls in optimizers_cls:
                                        for learning_rate in learning_rates:
                                            for loss in losses:
                                                row = {"model_name": "{}{}".format(model_cls.__name__, i),
                                                       "model_cls": model_cls,
                                                       "out_class": out_class,
                                                       "grayscale": grayscale,
                                                       "conv_sizes": conv_sizes,
                                                       "dense_sizes": dense_sizes,
                                                       "dropout": dropout,
                                                       "epochs": epochs,
                                                       "batch_size": batch_size,
                                                       "optimizer_cls": optimizer_cls,
                                                       "learning_rate": learning_rate,
                                                       "loss": loss}
                                                models_params.append(row)
                                                i += 1

    columns = ["model_name", "model_cls",
               "out_class",
               "grayscale",
               "conv_sizes",
               "dense_sizes",
               "dropout",
               "epochs",
               "batch_size",
               "optimizer_cls",
               "learning_rate",
               "loss",
               "epoch_time", "time",
               "mean_absolute_error", "mean_squared_error"]
    results = pd.DataFrame(columns=columns)

    dt = datetime.now().strftime('%Y-%m-%d-t%H-%M')
    for i in range(len(models_params)):
        model_params = models_params[i]
        train_dataset, test_dataset = load_hysts_mpiigaze_train_test_ds_generator(person=0,
                                                                                  val_split=0.2,
                                                                                  batch_size=model_params["batch_size"])

        optimizer = tf.keras.optimizers.get(model_params["optimizer_cls"]).from_config(
            {"learning_rate": model_params["learning_rate"]})
        loss = tf.keras.losses.get(model_params["loss"])

        model = model_params["model_cls"](conv_sizes=model_params["conv_sizes"],
                                          dense_sizes=model_params["dense_sizes"],
                                          dropout=model_params["dropout"])
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=[tf.keras.metrics.MeanAbsoluteError(),
                               tf.keras.metrics.MeanSquaredError()])
        start = time.time()
        history = model.fit_generator(generator=train_dataset, validation_data=test_dataset,
                                      epochs=model_params["epochs"])
        history = history.history
        elapsed_time = time.time() - start

        params = model_params.copy()
        row = {"model_name": params["model_name"],
               "model_cls": params["model_cls"].__name__,
               "out_class": params["out_class"],
               "grayscale": params["grayscale"],
               "conv_sizes": params["conv_sizes"],
               "dense_sizes": params["dense_sizes"],
               "dropout": params["dropout"],
               "epochs": params["epochs"],
               "batch_size": params["batch_size"],
               "optimizer_cls": params["optimizer_cls"],
               "learning_rate": params["learning_rate"],
               "loss": params["loss"],
               "epoch_time": elapsed_time / params["epochs"],
               "time": elapsed_time,
               "mean_absolute_error": history["mean_absolute_error"][-1],
               "mean_squared_error": history["mean_squared_error"][-1]}
        print(row)
        row = pd.DataFrame([row], columns=columns)
        results = results.append(row, ignore_index=True)

        # savings
        dirpath = os.path.join(RESULTS_PATH, "tables")
        results.to_csv("{}/results-{}.csv".format(dirpath, dt))
        plot_metrics(history, ["loss", "mean_absolute_error", "mean_squared_error"], params["model_name"])
        dirpath = os.path.join(RESULTS_PATH, "models")
        model.save_weights('{}/weights-{}-{}.h5'.format(
            dirpath, params["model_name"], dt))
        dirpath = os.path.join(RESULTS_PATH, "models_summaries")
        save_summary(model, dirpath)

        del model
        del train_dataset
        del test_dataset


if __name__ == '__main__':
    create_dirs()
    do_experiments()
