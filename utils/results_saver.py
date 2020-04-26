import json
import os

import numpy as np
import pandas as pd
from datetime import datetime
from comet_ml import Experiment

from settings import DATA_PATH, RESULTS_PATH
from utils.configs import USE_FLOAT64
from utils.metrics import compute_angle_error
from utils.plots import plot_metrics


def save_parameters_to_comet(experiment_name: str,
                             data_set,
                             person_id,
                             model_cls,
                             epochs,
                             conv_sizes,
                             dense_sizes,
                             dropout,
                             optimizer_name,
                             learning_rate,
                             loss_name
                             ):
    with open(os.path.join(DATA_PATH, "credentials/comet.json"), 'r') as f:
        credentials = json.load(f)
    experiment = Experiment(api_key=credentials['api_key'],
                            project_name=credentials['project_name'],
                            workspace=credentials['workspace'])
    experiment.set_name(experiment_name)
    _save_data_set_parameters_to_comet(experiment, data_set=data_set)
    experiment.log_parameter('person_id', person_id)
    experiment.log_parameter('model_cls', model_cls.__name__)
    experiment.log_parameter('epochs', epochs)
    experiment.log_parameter('conv_sizes', conv_sizes)
    experiment.log_parameter('dense_sizes', dense_sizes)
    experiment.log_parameter('dense_layers', dense_sizes)
    experiment.log_parameter('dropout', dropout)
    experiment.log_parameter('optimizer_name', optimizer_name)
    experiment.log_parameter('learning_rate', learning_rate)
    experiment.log_parameter('loss_name', loss_name)
    experiment.log_parameter('use_float64', USE_FLOAT64)

    return experiment


def save_metrics_to_comet(experiment, labels, predictions, test_subject_ids, forward_pass_time, use_gpu,
                          angle_error=True):
    if angle_error:
        angle_error = np.mean(compute_angle_error(labels=labels, predictions=predictions))
        experiment.log_metric("test_angle_error_degrees", angle_error)

        if test_subject_ids is not None:
            unique_subject_ids = np.unique(test_subject_ids)
            for id in unique_subject_ids:
                angle_error = np.mean(compute_angle_error(labels=labels[test_subject_ids == id],
                                                          predictions=predictions[test_subject_ids == id]))
                experiment.log_metric(f"test_angle_error_degrees_{id}", angle_error)

    if forward_pass_time is not None:
        host = "gpu" if use_gpu else "cpu"
        experiment.log_metric(f"forward_pass_time_{host}", forward_pass_time)


def _save_data_set_parameters_to_comet(experiment, data_set):
    for key, value in data_set.items():
        experiment.log_parameter(f"data_set_{key}", value)


# def save_results_locally(name: str, experiment, start_datetime: datetime, experiment_id, model, metrics, history):
#     dir_path = os.path.join(RESULTS_PATH, name)
#     start_datetime_str = start_datetime.strftime("%Y-%m-%d-t%H-%M-%S")
#     save_metrics_plots(dir_path, start_datetime_str, experiment_id, history, metrics)
#     save_table(dir_path, start_datetime_str, experiment_id, experiment)
#     save_weights(dir_path, model=model, start_datetime_str=start_datetime_str, experiment_id=experiment_id)


def save_table(dir_path: str, start_datetime_str: str, experiment_id, experiment):
    subdir_path = os.path.join(dir_path, "tables")
    subdir_path = os.path.join(subdir_path, start_datetime_str)

    columns = list(experiment.params.keys()) + list(experiment.metrics.keys())
    columns.sort()
    columns = ["experiment_id"] + columns
    row = dict()
    row["experiment_id"] = experiment_id
    for key, value in experiment.params.items():
        row[key] = value
    for key, value in experiment.metrics.items():
        if not key.startswith('sys'):
            row[key] = value

    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path, exist_ok=True)
    path = os.path.join(subdir_path, "results.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path, index_col=False)
        df = df.append([row], ignore_index=True)
        df.to_csv(path, index=False)
    else:
        df = pd.DataFrame([row], columns=columns)
        df.to_csv(path, index=False)


def save_weights(dir_path, start_datetime_str, experiment_id, model):
    subdir_path = os.path.join(dir_path, "models")
    subdir_path = os.path.join(subdir_path, start_datetime_str)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path, exist_ok=True)
    model.save_weights(os.path.join(subdir_path, 'weights-{}.h5'.format(str(experiment_id).zfill(4))))


def save_plots(dir_path, start_datetime_str, experiment_id, history, metrics):
    subdir_path = os.path.join(dir_path, "plots")
    subdir_path = os.path.join(subdir_path, start_datetime_str)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path, exist_ok=True)
    plot_metrics(subdir_path, str(experiment_id).zfill(4), history, metrics)
