# ## Plots
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
from contextlib import redirect_stdout
from datetime import datetime
from settings import DATA_PATH

RESULTS_PATH = os.path.join(DATA_PATH, "results")


def create_dirs():
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    dirs = ["plots", "models", "tables", "models_summaries"]
    for d in dirs:
        path = os.path.join(RESULTS_PATH, d)
        if not os.path.exists(path):
            os.makedirs(path)


def plot_metrics(history, metric_names, name):
    n_metrics = len(metric_names)

    for i, metric_name in enumerate(metric_names):
        train_metric = history[metric_name]
        val_metric = history[f'val_{metric_name}']

        epochs = list(range(1, len(train_metric) + 1))

        plt.figure(figsize=(30, 10))

        plt.subplot(i, n_metrics, 1)
        plt.plot(epochs, train_metric, '--o', label='train')
        plt.plot(epochs, val_metric, '--o', label='validation')
        plt.xlabel('num of epochs')
        plt.ylabel(metric_name)
        plt.legend()
        plt.title(metric_name)

    dirpath = os.path.join(RESULTS_PATH, "plots")

    plt.savefig('{}/{}-{}.png'.format(dirpath, name, datetime.now().strftime('%Y-%m-%d-t%H-%M')), bbox_inches='tight')


def prepare_and_plot_confusion_matrix(name, predictions, y_test, class_names):
    conf_matrix = tf.math.confusion_matrix(y_test, predictions)
    if tf.is_tensor(conf_matrix):
        conf_matrix = conf_matrix.numpy()
    conf_matrix = conf_matrix / conf_matrix.sum(axis=1)
    _plot_confusion_matrix(conf_matrix, name, class_names)


def _plot_confusion_matrix(confusion_matrix, name, class_names, figsize=(12, 12)):
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues")
    bottom, top = heatmap.get_ylim()
    heatmap.set_ylim(bottom + 0.5, top - 0.5)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    dirpath = os.path.join(RESULTS_PATH, "plots")

    plt.savefig('{}/{}-cmatrix-{}.png'.format(dirpath, name, datetime.now().strftime('%Y-%m-%d-t%H-%M')), bbox_inches='tight')


def save_summary(model, dir):
    with open('{}/{}-summary.txt'.format(dir, model.name), 'w') as f:
        with redirect_stdout(f):
            model.summary()


def load_model(model_params, model_weights_path, train_dataset, x_test, y_test):
    """
    Warning: It is required that model trains for one epoch before loading weights form file.
    """
    model = VggModel(model_params)
    model.fit(train_dataset=train_dataset, x_test=x_test, y_test=y_test, epochs=1)
    model.load_weights(filepath=model_weights_path)
    return model