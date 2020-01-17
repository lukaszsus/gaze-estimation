import tensorflow as tf


class Modal3ConvNet(tf.keras.Model):
    """
    Convolutional Neural Network with 3 modals:
        -   right eye image
        -   left eye image
        -   headpose
    """
    def __init__(self, height, width, num_channels, conv_sizes, dense_sizes, dropout):
        """Inits the class."""
        super().__init__()
        self.right_conv_layers = None
        self.left_conv_layers = None
        self._init_conv_layers(conv_sizes)
        self.flatten = None
        self._init_flatten()
        self.dense_layers = None
        self._init_dense_layers(dense_sizes, dropout)

    def _init_conv_layers(self, conv_sizes=None):
        if conv_sizes is None:
            conv_sizes = [{"n_filters": 16, "filter_size": (5, 5), "padding": "valid", "stride": (1, 1),
                           "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)},
                          {"n_filters": 16, "filter_size": (5, 5), "padding": "valid", "stride": (1, 1),
                           "pool": "avg", "pool_size": (2, 2), "pool_stride": (2, 2)}]
        self.right_conv_layers = []
        self.left_conv_layers = []
        for layers in (self.right_conv_layers, self.left_conv_layers):
            for i, layer in enumerate(conv_sizes):
                layers.append(tf.keras.layers.Conv2D(layer["n_filters"], layer["filter_size"],
                                                               activation='relu', padding=layer["padding"],
                                                               strides=layer["stride"]))
                if layer["pool"] == "avg":
                    layers.append(tf.keras.layers.AvgPool2D(pool_size=layer["pool_size"],
                                                            strides=layer["pool_stride"]))
                elif layer["pool"] == "max":
                    layers.append(tf.keras.layers.MaxPool2D(pool_size=layer["pool_size"],
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
        (x_right_eye, x_left_eye, x_headpose) = inputs
        # print(x_right_eye.get_shape())
        # print(x_left_eye.get_shape())
        # print(x_headpose.get_shape())

        # modal 1
        for conv_layer in self.right_conv_layers:
            x_right_eye = conv_layer(x_right_eye)
        # modal 2
        for conv_layer in self.left_conv_layers:
            x_left_eye = conv_layer(x_left_eye)

        # print(x_right_eye.get_shape())
        # print(x_left_eye.get_shape())
        # print(x_headpose.get_shape())

        # flattening and concatenating
        x_right_eye = self.flatten(x_right_eye)
        x_left_eye = self.flatten(x_left_eye)

        # modal 3
        # print(x_right_eye.get_shape())
        # print(x_left_eye.get_shape())
        # print(x_headpose.get_shape())
        x = tf.concat([x_right_eye, x_left_eye, x_headpose], 1)

        for dense_layer in self.dense_layers:
            x = dense_layer(x)

        return x

    def predict(self, x):
        """Predicts outputs based on inputs (x)."""
        return self.call(x)

