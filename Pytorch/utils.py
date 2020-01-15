## utils.py -- Some utility functions
##
## Copyright (C) 2018, IBM Corp
##                     Chun-Chen Tu <timtu@umich.edu>
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

## (C) 2020 Changes by UvA FACT AI group [Pytorch conversion]

from torch.nn import Module, Conv2d, LeakyReLU, MaxPool2d, Upsample, Sequential
from torch import load, save, eye, uint8, from_numpy, argmax
from torchsummary import summary
from torchvision.utils import save_image

import os
import h5py
import numpy as np
from PIL import Image


class AE(Module):
    def __init__(self, restore=None):
        """
        Autoencoder based on `mnist_AE_1_decoder.json`.
        """
        super(AE, self).__init__()

        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        self.kernel_size = (3,3)
        self.pool_kernel_size = (2,2)
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

def h5_to_state_dict(h5_file, mapping):
    """Create Pytorch state_dict from h5 weight with mapping."""
    state_dict = {}
    with h5py.File(h5_file, 'r') as f:
        for h5, state in mapping.items():
            state_dict[state] = from_numpy(f[h5][:].T)
    return state_dict

def load_AE(codec_prefix, print_summary=False, dir="models/"):
    """Load autoencoder from json. Optionally print summary"""
    # Weight file
    weight_file = dir + codec_prefix  + ".h5"
    if not os.path.isfile(weight_file):
        raise Exception(f"Decoder weight file {weight_file} not found.")

    # AE mapping
    AE_map = {
        'conv2d_4/conv2d_4/bias:0'       : 'decoder.0.bias',
        'conv2d_4/conv2d_4/kernel:0'     : 'decoder.0.weight',
        'conv2d_5/conv2d_5/bias:0'       : 'decoder.3.bias',
        'conv2d_5/conv2d_5/kernel:0'     : 'decoder.3.weight',
        'conv2d_6/conv2d_6/bias:0'       : 'decoder.5.bias',
        'conv2d_6/conv2d_6/kernel:0'     : 'decoder.5.weight',
        'sequential_1/conv2d_1/bias:0'   : 'encoder.0.bias',
        'sequential_1/conv2d_1/kernel:0' : 'encoder.0.weight',
        'sequential_1/conv2d_2/bias:0'   : 'encoder.2.bias',
        'sequential_1/conv2d_2/kernel:0' : 'encoder.2.weight',
        'sequential_1/conv2d_3/bias:0'   : 'encoder.5.bias',
        'sequential_1/conv2d_3/kernel:0' : 'encoder.5.weight',
    }

    # Create AE
    ae = AE(h5_to_state_dict(weight_file, AE_map))

    if print_summary:
        summary(ae, (ae.image_size, ae.image_size, ae.num_channels))

    return ae

def save_img(img, name="output.png"):
    """Save an MNIST image to location name, both as .pt and .png."""
    # Save tensor
    save(img, name+'.pt')

    # Save image, invert MNIST read
    fig = ((img + 0.5) * 255).round()
    fig = fig.type(uint8).squeeze()
    pic = Image.fromarray(fig.cpu().data.numpy())
    pic.save(name+'.png')

def generate_data(data, id, target_label):
    """
    Return test data id and one hot target.
    Expects data to be MNIST pytorch.
    """
    inputs = data.test_data[id]
    targets = eye(data.test_labels.shape[1])[target_label]

    return inputs, targets

def model_prediction(model, inputs):
    """
    Make a prediction for model given inputs.
    Returns: raw output, predicted class and raw output as string.
    """
    prob = model.predict(inputs)
    predicted_class = argmax(prob, dim=-1)
    prob_str = np.array2string(prob.cpu().data.numpy()).replace('\n','')

    return prob, predicted_class.item(), prob_str

# Example
# ae = load_AE('mnist_AE_weights', True)
