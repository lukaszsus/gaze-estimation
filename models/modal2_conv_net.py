import time

import tensorflow as tf
from tqdm import tqdm

from utils.metrics import compute_angle_error


class Modal2ConvNet(tf.keras.Model):
    """
    Convolutional Neural Network with 2 modals:
        -   eye image
        -   headpose
    """

    def __init__(self, conv_sizes, dense_sizes, dropout, output_size, track_angle_error):
        """
        :track_angle_error: True cause that every test step will be done twice.
                            It will count angle error and send it to comet.ml.
        """
        super().__init__()
        self.conv_layers = None
        self._init_conv_layers(conv_sizes)
        self.flatten = None
        self._init_flatten()
        self.dense_layers = None
        self._init_dense_layers(dense_sizes, dropout)
        self.output_size = output_size
        self.track_angle_error = track_angle_error

    def _init_conv_layers(self, conv_sizes=None):
        if conv_sizes is None:
            conv_sizes = [{"n_filters": 16, "filter_size": (5, 5), "padding": "valid", "stride": (1, 1),
                           "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)},
                          {"n_filters": 16, "filter_size": (5, 5), "padding": "valid", "stride": (1, 1),
                           "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)}]
        self.conv_layers = []
        for i, layer in enumerate(conv_sizes):
            self.conv_layers.append(tf.keras.layers.Conv2D(layer["n_filters"], layer["filter_size"],
                                                           activation='relu', padding=layer["padding"],
                                                           strides=layer["stride"]))
            if layer["pool"] == "avg":
                self.conv_layers.append(tf.keras.layers.AvgPool2D(pool_size=layer["pool_size"],
                                                                  strides=layer["pool_stride"]))
            elif layer["pool"] == "max":
                self.conv_layers.append(tf.keras.layers.MaxPool2D(pool_size=layer["pool_size"],
                                                                  strides=layer["pool_stride"]))

    def _init_flatten(self):
        self.flatten = tf.keras.layers.Flatten()

    def _init_dense_layers(self, dense_sizes=None, dropout=0.0):
        if dense_sizes is None:
            dense_sizes = [64, 16, 2]
        self.dense_layers = []
        for i, size in enumerate(dense_sizes):
            if i < len(dense_sizes) - 1:
                self.dense_layers.append(tf.keras.layers.Dense(size, activation=tf.nn.relu,
                                                               kernel_initializer="glorot_normal"))
                self.dense_layers.append(tf.keras.layers.Dropout(dropout))
            else:
                self.dense_layers.append(tf.keras.layers.Dense(size, kernel_initializer="glorot_normal"))

    def call(self, inputs, training=False):
        """Makes forward pass of the network."""
        (x_eye, x_headpose) = inputs

        # modal 1
        for conv_layer in self.conv_layers:
            x_eye = conv_layer(x_eye)

        # flattening and concatenating
        x_eye = self.flatten(x_eye)

        x = tf.concat([x_eye, x_headpose], 1)

        for dense_layer in self.dense_layers:
            x = dense_layer(x)

        return x

    def predict(self, x):
        """Predicts outputs based on inputs (x)."""
        return self.call(x)

    @tf.function
    def train_step(self, examples, labels):
        # what should tf count gradient for
        with tf.GradientTape() as tape:
            predictions = self(examples, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_mae.update_state(labels, predictions)
        self.train_mse.update_state(labels, predictions)
        for i in range(self.output_size):
            self.train_mae_column[i].update_state(labels[:, i], predictions[:, i])

    @tf.function
    def test_step(self, examples, labels):
        predictions = self(examples)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss.update_state(t_loss)
        self.test_mae.update_state(labels, predictions)
        self.test_mse.update_state(labels, predictions)
        for i in range(self.output_size):
            self.test_mae_column[i].update_state(labels[:, i], predictions[:, i])

    def compile(self, **kwargs):
        optimizer = kwargs.get("optimizer", 'adam')
        loss = kwargs.get("loss", "mae")
        self.__parse_optimizer(optimizer)
        self.__parse_loss_object(loss)

    def fit(self, **kwargs):
        """
        Implements learning loop for the model.
        kwargs can contain optional parameters such as
        num_epochs, batch_size, etc.
        """
        experiment = kwargs['experiment']
        self.train_dataset = kwargs.get('train_dataset', None)
        self.test_dataset = kwargs.get('test_dataset', None)
        epochs = kwargs.get('epochs', 10)

        # metrics
        self.__create_metrics_history()
        self.__create_metrics()

        start_time = time.time()
        for epoch in range(epochs):
            if experiment is not None:
                experiment.set_epoch(epoch)
                experiment.set_step(epoch)

            for examples, labels in tqdm(self.train_dataset):
                self.train_step(examples, labels)

            for x_test, y_test in tqdm(self.test_dataset):
                self.test_step(x_test, y_test)

            template = 'Epoch: {}, Loss: {}, Train MAE: {}, Train MSE: {}, Test Loss: {}, Test MAE: {}, Test MSE: {}'
            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.train_mae.result(),
                                  self.train_mse.result(),
                                  self.test_loss.result(),
                                  self.test_mae.result(),
                                  self.test_mse.result()))

            if experiment is not None:
                self.__update_experiment(experiment, epoch)

            self.__update_metrics_history()
            # Reset the metrics for the next epoch
            self.__reset_metrics_states()

        elapsed = time.time() - start_time
        epoch_time = elapsed / epochs
        self.history["elapsed"] = elapsed
        self.history["epoch_time"] = epoch_time

        return self.history

    def __parse_loss_object(self, loss_object):
        if type(loss_object) == str:
            self.loss_object = tf.keras.losses.get(loss_object)
        elif loss_object is not None:
            self.loss_object = loss_object
        else:
            self.loss_object = tf.keras.losses.MeanAbsoluteError()

    def __parse_optimizer(self, optimizer):
        if type(optimizer) == str:
            self.optimizer = tf.keras.optimizers.get(optimizer)
        elif optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = tf.keras.optimizers.Adam()

    def __create_metrics(self):
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_mae = tf.keras.metrics.MeanAbsoluteError(name="train_mae")
        self.train_mse = tf.keras.metrics.MeanSquaredError(name="train_mse")
        self.train_mae_column = list()
        for column in range(self.output_size):
            self.train_mae_column.append(tf.keras.metrics.MeanAbsoluteError(name=f"train_mae_{column}"))
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_mae = tf.keras.metrics.MeanAbsoluteError(name="test_mae")
        self.test_mse = tf.keras.metrics.MeanSquaredError(name="test_mse")
        self.test_mae_column = list()
        for column in range(self.output_size):
            self.test_mae_column.append(tf.keras.metrics.MeanAbsoluteError(name=f"test_mae_{column}"))

    def __update_experiment(self, experiment, epoch):
        experiment.log_metric("train_loss", self.train_loss.result().numpy(), step=epoch)
        experiment.log_metric("train_mean_absolute_error", self.train_mae.result().numpy(), step=epoch)
        experiment.log_metric("train_mean_squared_error", self.train_mse.result().numpy(), step=epoch)
        for column in range(self.output_size):
            experiment.log_metric(f"train_mae_{column}", self.train_mae_column[column].result().numpy(), step=epoch)
        experiment.log_metric("test_loss", self.test_loss.result().numpy(), step=epoch)
        experiment.log_metric("test_mean_absolute_error", self.test_mae.result().numpy(), step=epoch)
        experiment.log_metric("test_mean_squared_error", self.test_mse.result().numpy(), step=epoch)
        for column in range(self.output_size):
            experiment.log_metric(f"test_mae_{column}", self.test_mae_column[column].result().numpy(), step=epoch)

        if self.track_angle_error:
            predictions = list()
            labels = list()
            for x, y in self.test_dataset:
                predictions.append(self.predict(x))
                labels.append(y)
            predictions = tf.concat(predictions, axis=0)
            labels = tf.concat(labels, axis=0)
            angle_error = tf.math.reduce_mean(compute_angle_error(labels=labels, predictions=predictions))
            experiment.log_metric("angle_error_degrees", angle_error, step=epoch)

    def __reset_metrics_states(self):
        self.train_loss.reset_states()
        self.train_mae.reset_states()
        self.train_mse.reset_states()
        for column in range(self.output_size):
            self.train_mae_column[column].reset_states()
        self.test_loss.reset_states()
        self.test_mae.reset_states()
        self.test_mse.reset_states()
        for column in range(self.output_size):
            self.test_mae_column[column].reset_states()

    def __create_metrics_history(self):
        self.history = {"train_loss": list(),
                        "train_mean_absolute_error": list(),
                        "train_mean_squared_error": list(),
                        "test_loss": list(),
                        "test_mean_absolute_error": list(),
                        "test_mean_squared_error": list(),
                        "elapsed": None,
                        "epoch_time": None}
        for column in range(self.output_size):
            self.history[f"train_mean_absolute_error_{column}"] = list()
        for column in range(self.output_size):
            self.history[f"test_mean_absolute_error_{column}"] = list()

    def __update_metrics_history(self):
        self.history["train_loss"].append(self.train_loss.result().numpy())
        self.history["train_mean_absolute_error"].append(self.train_mae.result().numpy())
        self.history["train_mean_squared_error"].append(self.train_mse.result().numpy())
        for column in range(self.output_size):
            self.history[f"train_mean_absolute_error_{column}"].append(self.train_mae_column[column].result().numpy())
        self.history["test_loss"].append(self.test_loss.result().numpy())
        self.history["test_mean_absolute_error"].append(self.test_mae.result().numpy())
        self.history["test_mean_squared_error"].append(self.test_mse.result().numpy())
        for column in range(self.output_size):
            self.history[f"test_mean_absolute_error_{column}"].append(self.train_mae_column[column].result().numpy())

