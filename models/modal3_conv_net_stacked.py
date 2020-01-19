import tensorflow as tf

from models.modal3_conv_net import Modal3ConvNet


class Modal3ConvNetStacked(Modal3ConvNet):
    """
    Convolutional Neural Network with 3 modals:
        -   right eye image
        -   left eye image
        -   headpose
    """
    def __init__(self, conv_sizes, dense_sizes, dropout, output_size, track_angle_error):
        """Inits the class."""
        super().__init__(conv_sizes, dense_sizes, dropout, output_size, track_angle_error)

    def _init_conv_layers(self, conv_sizes=None):
        if conv_sizes is None:
            conv_sizes = [{"n_filters": 16, "filter_size": (5, 5), "padding": "valid", "stride": (1, 1),
                           "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)},
                          {"n_filters": 16, "filter_size": (5, 5), "padding": "valid", "stride": (1, 1),
                           "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)}]
        self.right_conv_layers = []
        for i, layer in enumerate(conv_sizes):
            self.right_conv_layers.append(tf.keras.layers.Conv2D(layer["n_filters"], layer["filter_size"],
                                                           activation='relu', padding=layer["padding"],
                                                           strides=layer["stride"]))
            if layer["pool"] == "avg":
                self.right_conv_layers.append(tf.keras.layers.AvgPool2D(pool_size=layer["pool_size"],
                                                        strides=layer["pool_stride"]))
            elif layer["pool"] == "max":
                self.right_conv_layers.append(tf.keras.layers.MaxPool2D(pool_size=layer["pool_size"],
                                                        strides=layer["pool_stride"]))

    def call(self, inputs, training=False):
        """Makes forward pass of the network."""
        (x_right_eye, x_left_eye, x_headpose) = inputs

        # modal 1 and 2
        x_eye = tf.concat((x_right_eye, x_left_eye), axis=-1)
        for conv_layer in self.right_conv_layers:
            x_eye = conv_layer(x_eye)

        # flattening and concatenating
        x_eye = self.flatten(x_eye)

        # modal 3
        x = tf.concat([x_eye, x_headpose], 1)

        for dense_layer in self.dense_layers:
            x = dense_layer(x)

        return x

