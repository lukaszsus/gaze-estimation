import json
import os
import pickle
import comet_ml
from settings import PROJECT_PATH
from tqdm import tqdm
import pandas as pd
from datetime import datetime


def get_all_experiments():
    with open(f"{PROJECT_PATH}/data/credentials/comet.json", 'r') as f:
        credentials = json.load(f)

    api_key = credentials['api_key']
    project_name = credentials['project_name']
    workspace = credentials['workspace']

    comet_api = comet_ml.api.API(rest_api_key=api_key)
    experiments_original = comet_api.get_experiments(workspace, project_name=project_name)

    dt = datetime.now().strftime('%Y-%m-%d-t%H-%M')
    file_path = f'{PROJECT_PATH}/experiments_files/experiments-{dt}.pickle'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            experiments: pd.DataFrame = pickle.load(f)
    else:
        experiments = pd.DataFrame(columns=["id"])

    for experiment in tqdm(experiments_original):
        if experiment.id not in experiments['id'].tolist():
            new_experiment = {'name': experiment.name, 'id': experiment.id}
            for param in experiment.get_parameters_summary():
                new_experiment[param['name']] = param['valueCurrent']

            for metric in experiment.get_metrics_summary():
                new_experiment[metric['name']] = metric['valueCurrent']

            for other in experiment.get_others_summary():
                if other['name'] == 'Name':
                    new_experiment['name'] = other['valueCurrent']

            experiments = experiments.append(new_experiment, ignore_index=True)

    with open(file_path, 'wb') as f:
        pickle.dump(experiments, f)


# def get_experiments():
#     name = "mpiigaze_one_eye_leave_one_out"
#
#     with open(f"{PROJECT_PATH}/data/credentials/comet.json", 'r') as f:
#         credentials = json.load(f)
#
#     api_key = credentials['api_key']
#     project_name = credentials['project_name']
#     workspace = credentials['workspace']
#
#     comet_api = comet_ml.api.API(rest_api_key=api_key)
#     # experiments_original = comet_api.get_experiments("lukaszsus/gaze-estimation/one_eye_leave_one_out")
#     experiments_original = comet_api.get_experiments(project_name, workspace=workspace, pattern=name)
#
#     file_path = f'{PROJECT_PATH}/experiments_files/{name}.pickle'
#     if os.path.exists(file_path):
#         with open(file_path, 'rb') as f:
#             experiments: pd.DataFrame = pickle.load(f)
#     else:
#         experiments = pd.DataFrame(columns=["id"])
#
#     for experiment in tqdm(experiments_original):
#         if experiment.id not in experiments['id'].tolist():
#             new_experiment = {'name': '', 'id': experiment.id}
#             for param in experiment.get_parameters_summary():
#                 new_experiment[param['name']] = param['valueCurrent']
#
#             for metric in experiment.get_metrics_summary():
#                 new_experiment[metric['name']] = metric['valueCurrent']
#
#             for other in experiment.get_others_summary():
#                 if other['name'] == 'Name':
#                     new_experiment['name'] = other['valueCurrent']
#
#             experiments = experiments.append(new_experiment, ignore_index=True)
#
#     with open(file_path, 'wb') as f:
#         pickle.dump(experiments, f)


if __name__ == "__main__":
    get_all_experiments()