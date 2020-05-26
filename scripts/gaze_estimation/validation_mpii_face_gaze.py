from data_loader.validation_with_loaded_model import perform_test_with_loaded_data
from models.model_loaders import load_best_modal3_conv_net
from utils.datasets import data_sets
from models.modal3_conv_net import Modal3ConvNet
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


if __name__ == "__main__":
    validation_mpii_face_gaze()
