# models.py -- MNIST data and model loading code to create model classes.

# (C) 2020 Changes by UvA FACT AI group [Pytorch conversion]

# Based on:
# Copyright (C) 2018, IBM Corp
#                     Chun-Chen Tu <timtu@umich.edu>
#                     PaiShun Ting <paishun@umich.edu>
#                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>

from torch.nn import Sequential, Conv2d, LeakyReLU, MaxPool2d, Flatten, \
                     Linear, Softmax, Module, Upsample


class AE(Module):
    def __init__(self, restore=None):
        """
        Autoencoder architecture based on mnist_AE_1_decoder.json.
        """

        super(AE, self).__init__()

        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        self.kernel_size = (3, 3)
        self.pool_kernel_size = (2, 2)
        self.relu_slope = 0
        self.filter_size = 16
        self.padding = 1

        self.encoder = Sequential(
            Conv2d(self.num_channels, self.filter_size, self.kernel_size,
                   padding=self.padding),
            LeakyReLU(self.relu_slope),
            Conv2d(self.filter_size, self.filter_size, self.kernel_size,
                   padding=self.padding),
            LeakyReLU(self.relu_slope),
            MaxPool2d(self.pool_kernel_size),
            Conv2d(self.filter_size, self.num_channels, self.kernel_size,
                   padding=self.padding),
        )

        self.decoder = Sequential(
            Conv2d(self.num_channels, self.filter_size, self.kernel_size,
                   padding=self.padding),
            LeakyReLU(self.relu_slope),
            Upsample(scale_factor=self.pool_kernel_size),
            Conv2d(self.filter_size, self.filter_size, self.kernel_size,
                   padding=self.padding),
            LeakyReLU(self.relu_slope),
            Conv2d(self.filter_size, self.num_channels, self.kernel_size,
                   padding=self.padding),
        )

        # Load pre-trained weights.
        if restore:
            self.load_state_dict(restore)

    def predict(self, data):
        """Predict output for input data (batch, dim1, dim2, c)."""

        assert data[0].shape == (28, 28, 1), "Expected shape (b, 28, 28, 1)."

        # Reshape data, expect (batch, channel, dim1, dim2)
        data = data.view(-1, 1, self.image_size, self.image_size)
        latent = self.encoder(data)
        pred = self.decoder(latent)

        return pred.view(-1, self.image_size, self.image_size, 1)

    def forward(self, data):
        """Predict alias."""

        return self.predict(data)


class MNISTModel(Module):
    def __init__(self, restore=None, use_log=False):
        """
        Init MNISTModel with a Convolutional Neural Network.

        Arguments:
        - restore: supply a loaded Pytorch state dict to reload weights
        - use_log: bool: output log probability for attack
        """
        super(MNISTModel, self).__init__()

        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        self.kernel_size = (3, 3)
        self.pool_kernel_size = (2, 2)
        self.relu_slope = 0

        self.output12 = 32
        self.output34 = 64
        self.output56 = 200
        self.output_flat = 1024

        model = [
            Conv2d(self.num_channels, self.output12, self.kernel_size),
            LeakyReLU(self.relu_slope),
            Conv2d(self.output12, self.output12, self.kernel_size),
            LeakyReLU(self.relu_slope),
            MaxPool2d(self.pool_kernel_size),
            #
            Conv2d(self.output12, self.output34, self.kernel_size),
            LeakyReLU(self.relu_slope),
            Conv2d(self.output34, self.output34, self.kernel_size),
            LeakyReLU(self.relu_slope),
            MaxPool2d(self.pool_kernel_size),
            #
            Flatten(),
            Linear(self.output_flat, self.output56),
            LeakyReLU(self.relu_slope),
            Linear(self.output56, self.output56),
            LeakyReLU(self.relu_slope),
            Linear(self.output56, self.num_labels)
        ]

        # Softmax activation probability, only sed for black box attacks.
        if use_log:
            model += [Softmax(dim=-1)]

        self.model = Sequential(*model)

        # Load pre-trained weights.
        if restore:
            self.load_state_dict(restore)

    def predict(self, data):
        """
        Predict output of MNISTModel for input data (batch, dim1, dim2, c).
        """
        assert data[0].shape == (28, 28, 1), "Expected shape (28, 28, 1)."

        # Reshape data for batch size, expect (batch, channel, dim1, dim2)
        return self.model(data.view(-1, 1, self.image_size, self.image_size))

    def forward(self, data):
        """Predict alias."""
        return self.predict(data)
