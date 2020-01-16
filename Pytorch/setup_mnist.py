## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

# import tensorflow as tf

# (C) 2020 Changes by UvA FACT AI group [Pytorch conversion]

import numpy as np
import os
import pickle
import gzip
import urllib.request

from torch.nn import Sequential, Conv2d, LeakyReLU, MaxPool2d, Flatten, Linear,\
                     Softmax, Module
from torch import from_numpy, load
import sys
from torchsummary import summary

def extract_data(filename, num_images, img_size=28):
    """Read MNIST image file as pytorch tensor."""
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*img_size*img_size)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, img_size, img_size, 1)
    return from_numpy(data)

def extract_labels(filename, num_images):
    """Read MNIST label file as pytorch tensor."""
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return from_numpy((np.arange(10) == labels[:, None]).astype(np.float32))

class MNIST:
    def __init__(self, force=False):
        """Load MNIST dataset, optionally force to download and overwrite."""
        self.n_train = 60000
        self.n_test  = 10000
        self.n_valid = 5000

        self.force = force

        # Make storage room
        self.dir = "data"
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
            self.force = True

        # Get files locally
        self.mnist_url = "http://yann.lecun.com/exdb/mnist/"
        self.train_x_path = self.fetch("train-images-idx3-ubyte.gz")
        self.train_r_path = self.fetch("train-labels-idx1-ubyte.gz")
        self.test_x_path = self.fetch("t10k-images-idx3-ubyte.gz")
        self.test_r_path = self.fetch("t10k-labels-idx1-ubyte.gz")

        # Get Test data
        self.test_data = extract_data(self.test_x_path, self.n_test)
        self.test_labels = extract_labels(self.test_r_path, self.n_test)

        # Get Train data
        train_data = extract_data(self.train_x_path, self.n_train)
        train_labels = extract_labels(self.train_r_path, self.n_train)
        self.train_data = train_data[self.n_valid:, :, :, :]
        self.train_labels = train_labels[self.n_valid:]

        # Get Validation data from training data
        self.validation_data = train_data[:self.n_valid, :, :, :]
        self.validation_labels = train_labels[:self.n_valid]

    def fetch(self, file):
        """Get file from self.url if not already present locally."""
        path = self.dir+"/"+file
        if self.force or not os.path.exists(path):
            urllib.request.urlretrieve(self.mnist_url+file, path)
        return path

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

        self.kernel_size = (3,3)
        self.pool_kernel_size = (2,2)
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

        # output log probability, used for black-box attack
        if use_log:
            model += [Softmax(dim=-1)]

        self.model = Sequential(*model)

        if restore:
            self.load_state_dict(restore)

    def predict(self, data):
        """Predict output of MNISTModel for input data (batch, dim1, dim2, c)."""
        # print(data.shape)
        # assert data[0].shape == (28, 28, 1), "Expected shape (28, 28, 1)."

        # Reshape data, expect (batch, channel, dim1, dim2)
        # return self.model(data.view(-1, 1, self.image_size, self.image_size))
        return self.model(data.permute(0,3,1,2))

    def forward(self, data):
        """alias."""
        return self.predict(data)

# import torch
# data = MNIST()
# m = MNISTModel(torch.load('models/mnist.pt'))
# ps = m.predict(data.test_data[0:10])
# print(ps.shape)
