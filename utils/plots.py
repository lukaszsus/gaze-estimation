import matplotlib.pyplot as plt
import os
from settings import RESULTS_PATH


def create_dirs():
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    dirs = ["plots", "models", "tables", "models_summaries"]
    for d in dirs:
        path = os.path.join(RESULTS_PATH, d)
        if not os.path.exists(path):
            os.makedirs(path)


def plot_metrics(dirpath, name, history, metric_names):
    n_metrics = len(metric_names)

    plt.figure(figsize=(n_metrics * 10, 10))

    for i, metric_name in enumerate(metric_names):
        train_metric = history[f'train_{metric_name}']
        val_metric = history[f'test_{metric_name}']

        epochs = list(range(1, len(train_metric) + 1))

        plt.subplot(1, n_metrics, i + 1)
        plt.plot(epochs, train_metric, '--o', label='train')
        plt.plot(epochs, val_metric, '--o', label='validation')
        plt.xlabel('num of epochs')
        plt.ylabel(metric_name)
        plt.legend()
        plt.title(metric_name)

    plt.savefig('{}/{}.png'.format(dirpath, name, bbox_inches='tight'))