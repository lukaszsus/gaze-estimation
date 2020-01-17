import json
import os
import pandas as pd
from datetime import datetime
from comet_ml import Experiment

from settings import DATA_PATH, RESULTS_PATH
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

    return experiment


def _save_data_set_parameters_to_comet(experiment, data_set):
    for key, value in data_set.items():
        experiment.log_parameter(f"data_set_{key}", value)


def save_results_locally(name: str, experiment, start_time: datetime, experiment_id, model, metrics, history):
    dir_path = os.path.join(RESULTS_PATH, name)
    start_time_str = start_time.strftime("%Y-%m-%d-t%H-%M-%S")
    _plot_metrics(dir_path, start_time_str, experiment_id, history, metrics)
    _save_table(dir_path, start_time_str, experiment_id, experiment)
    _save_weights(dir_path, model=model, start_time_str=start_time_str, experiment_id=experiment_id)


def _save_table(dir_path: str, start_time_str: str, experiment_id, experiment):
    subdir_path = os.path.join(dir_path, "tables")
    subdir_path = os.path.join(subdir_path, start_time_str)

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


def _save_weights(dir_path, start_time_str, experiment_id, model):
    subdir_path = os.path.join(dir_path, "models")
    subdir_path = os.path.join(subdir_path, start_time_str)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path, exist_ok=True)
    model.save_weights(os.path.join(subdir_path, 'weights-{}.h5'.format(str(experiment_id).zfill(4))))


def _plot_metrics(dir_path, start_time_str, experiment_id, history, metrics):
    subdir_path = os.path.join(dir_path, "plots")
    subdir_path = os.path.join(subdir_path, start_time_str)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path, exist_ok=True)
    plot_metrics(subdir_path, str(experiment_id).zfill(4), history, metrics)
