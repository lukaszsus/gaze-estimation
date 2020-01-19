import time

import tensorflow as tf
from tqdm import tqdm
from utils.metrics import compute_angle_error


class BaseRegressionModel(tf.keras.Model):
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
            experiment.log_metric("test_angle_error_degrees", angle_error, step=epoch)

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

