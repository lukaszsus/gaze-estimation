import os

from sklearn.metrics import mean_absolute_error

from models.modal3_conv_net import Modal3ConvNet
from settings import DATA_PATH
from utils.datasets import data_sets
from utils.metrics import final_predictions

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


def load_best_modal3_conv_net(test=False):
    # model params
    conv_sizes = ({"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                   "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)},
                  {"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                   "pool": None, "pool_size": None, "pool_stride": None},
                  {"n_filters": 16, "filter_size": (3, 3), "padding": "valid", "stride": (1, 1),
                   "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)})
    dense_sizes = (512, 128, 2)
    dropout = 0.1
    output_size = 2
    track_angle_error = True

    # learning params
    optimizer_name = "Adam"
    learning_rate = 0.001
    loss_name = 'mae'

    # dataset params
    data_set_name = 'mpiigaze_both_landmarks_coords_rgb'
    person_id = 0
    val_split = 0.05
    batch_size = 128

    # model initialization
    model = Modal3ConvNet(conv_sizes=conv_sizes, dense_sizes=dense_sizes, dropout=dropout,
                      output_size=output_size, track_angle_error=track_angle_error)
    optimizer = tf.keras.optimizers.get(optimizer_name).from_config({"learning_rate": learning_rate})
    loss = tf.keras.losses.get(loss_name)
    model.compile(optimizer=optimizer,
                  loss=loss)

    data_set = data_sets[data_set_name]
    data_set_loader = data_set["load_function"]
    train_dataset, test_dataset, test_subject_ids = data_set_loader(person_id=person_id,
                                                                    val_split=val_split,
                                                                    batch_size=batch_size)

    history = model.fit(experiment=None, train_dataset=train_dataset, test_dataset=test_dataset, epochs=1)

    weights_path = os.path.join(DATA_PATH, "models", "gaze_estimation", "best_modal3_conv_net.h5")
    model.load_weights(weights_path)

    if test:
        labels, predictions = final_predictions(model, test_dataset)
        print(f"MAE: {mean_absolute_error(labels, predictions)}")

    return model
