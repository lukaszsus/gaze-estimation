from utils.results_saver import save_parameters_to_comet
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_loader.mpiigaze_both_from_single import _load_all_people_reject_suspicious
from data_loader.mpiigaze_processed_loader import prepare_dataset
from utils.metrics import final_predictions


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


def load_test_own_mpiigaze_full(dataset_name, person_id, val_split=0.2, batch_size=128,
                                                 grayscale=True, all_subjects=None):
    if all_subjects is None:
        all_subjects = list(range(0, 7)) + list(range(8, 15)) + [24, 25] + list(range(30, 36))

    right_images, left_images, poses, gazes, subject_ids = _load_all_people_reject_suspicious(dataset_name,
                                                                                              grayscale,
                                                                                              subjects=all_subjects)
    indices = np.isin(subject_ids, all_subjects)

    right_images = right_images[indices]
    left_images = left_images[indices]
    poses = poses[indices]
    gazes = gazes[indices]
    test_subject_ids = subject_ids[indices]

    # don't shuffle test dataset to have consistent outcomes
    test_dataset = prepare_dataset((right_images, left_images, poses), gazes, batch_size, shuffle=False)

    return test_dataset, test_subject_ids