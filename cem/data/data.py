# data.py -- Prepare MNIST data and model loading code.
#
# Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
#
# This program is licenced under the BSD 2-Clause licence,
# contained in the LICENCE file in this directory.

# (C) 2020 Changes by UvA FACT AI group [Pytorch conversion]

import os
import gzip
import urllib.request
import numpy as np
from torch import from_numpy


class MNIST:
    def __init__(self, dvc='cpu', data_type='MNIST', force=False):
        """Load MNIST dataset, optionally force to download and overwrite."""
        # Create storage room.
        self.data_type = type
        self.force = force
        self.dir = os.path.dirname(__file__) + '/' + type

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            self.force = True

        # Retrieve MNIST files locally.
        self.url = \
            "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        if data_type == 'MNIST':
            self.url = "http://yann.lecun.com/exdb/mnist/"
        elif data_type != 'FMNIST':
            raise f"Unkown dataset type {data_type}"

        n_train = 60000
        n_test = 10000
        train_x_path = self.fetch("train-images-idx3-ubyte.gz")
        train_r_path = self.fetch("train-labels-idx1-ubyte.gz")
        test_x_path = self.fetch("t10k-images-idx3-ubyte.gz")
        test_r_path = self.fetch("t10k-labels-idx1-ubyte.gz")

        # Extract train and test data.
        self.train_data = MNIST.extract_data(train_x_path, n_train).to(dvc)
        self.train_labels = MNIST.extract_labels(train_r_path, n_train).to(dvc)
        self.test_data = MNIST.extract_data(test_x_path, n_test).to(dvc)
        self.test_labels = MNIST.extract_labels(test_r_path, n_test).to(dvc)

    def fetch(self, file):
        """Get file from self.url if not already present locally."""
        path = self.dir+"/"+file
        if self.force or not os.path.exists(path):
            url = self.url+file
            print(f"Downloading from {url} to {path}...", end=' ')
            urllib.request.urlretrieve(url, path)
            print('done')

        return path

    def extract_data(filename, num_images, img_size=28):
        """Read MNIST image file as pytorch tensor."""
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(num_images*img_size*img_size)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

            # Normalize data and center at zero.
            data = (data / 255) - 0.5
            data = data.reshape(num_images, img_size, img_size, 1)

        return from_numpy(data)

    def extract_labels(filename, num_images):
        """Read MNIST label file as pytorch tensor."""
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8)
        labels = (np.arange(10) == labels[:, None]).astype(np.float32)
        return from_numpy(labels)
