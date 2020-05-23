from utils.results_saver import save_metrics_to_comet, \
    save_parameters_to_comet
from models.model_loaders import load_best_modal3_conv_net
from utils.datasets import data_sets
from utils.metrics import final_test_measure_time, final_predictions
from models.modal3_conv_net import Modal3ConvNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

"""
That's for GPU training and maintaining one session and nice cuda lib loading.
"""
USE_GPU = True
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
AUTOTUNE = tf.data.experimental.AUTOTUNE


def perform_test_with_loaded_data(experiment_name,
                                  test_dataset,
                                  test_subject_ids,
                                  data_set,
                                  model,
                                  model_cls):
    experiment = save_parameters_to_comet(experiment_name, data_set=data_set, person_id=None,
                                          model_cls=model_cls,
                                          epochs=None,
                                          conv_sizes=None,
                                          dense_sizes=None,
                                          dropout=None,
                                          optimizer_name=None,
                                          learning_rate=None,
                                          loss_name=None)

    # final metrics
    labels, predictions = final_predictions(model, test_dataset)
    count_save_metrics_to_comet(experiment, labels=labels, predictions=predictions, test_subject_ids=test_subject_ids)

    del model
    experiment.end()


def validation_mpii_face_gaze():
    # hyper parameters
    experiment_name = 'mpii_face_gaze_own_pipeline'
    experiment_id = 0
    experiments_data_set_names = ['test_mpii_face_gaze_own_pipeline']
    batch_size = 128

    model = load_best_modal3_conv_net(test=False, file_name="modal3_conv_net_own_24_25.h5")

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


def count_save_metrics_to_comet(experiment, labels, predictions, test_subject_ids):
    mae = mean_absolute_error(labels.numpy(), predictions.numpy())
    mse = mean_squared_error(labels.numpy(), predictions.numpy())

    experiment.log_metric("test_mean_absolute_error", mae, step=1)
    experiment.log_metric("test_mean_squared_error", mse, step=1)
    for column in range(2):
        experiment.log_metric(f"test_mae_{column}", mean_absolute_error(labels[:, column].numpy(),
                                                                        predictions[:, column].numpy()), step=1)

    unique_subject_ids = np.unique(test_subject_ids)
    for id in unique_subject_ids:
        mae = mean_absolute_error(labels[test_subject_ids == id],
                                  predictions[test_subject_ids == id])
        experiment.log_metric(f"test_mae_person_{id}", mae)


if __name__ == "__main__":
    validation_mpii_face_gaze()
