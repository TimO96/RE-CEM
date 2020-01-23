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
from PIL import Image

import os
import h5py
import numpy as np

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

def save_img(img, name="output", channel=None, mode_img=None, save_tensor=True,
             thres=10, intensity=None):
    """Save an MNIST image to location name, both as .pt and .png."""
    # Save tensor
    if save_tensor:
        save(img, name+'.pt')

    # Save image, invert MNIST read
    fig = np.around((img.cpu().data.numpy() + 0.5) * 255)
    fig = fig.astype(np.uint8).squeeze()

    if channel:
        channel = 1 if (channel == 'PP') else 0
        nfig = np.zeros((3, *fig.shape)).astype(np.uint8)
        if mode_img is not None:
            mode_img = np.around((mode_img.cpu().data.numpy() + 0.5) * 255)
            mode_img = mode_img.astype(np.uint8).squeeze()

            # Thresholding tricks.
            fig[mode_img > thres] = 0
            mode_img[mode_img <= thres] = 0
            if intensity:
                mode_img[mode_img > thres] = intensity

            # Convert overlay to RGB.
            nfig[channel] = mode_img
            overlay = nfig.transpose(1,2,0)
        else:
            nfig[channel] = fig
            fig = nfig.transpose(1,2,0)

    pic = Image.fromarray(fig)

    # Add overlay.
    if mode_img is not None:
        pic = np.array(pic.convert('RGB'))
        pic += overlay
        pic = Image.fromarray(pic)

    # name = 'output'
    pic.save(name+'.png')
    return pic

def generate_data(data, id, target_label):
    """
    Return test data id and one hot target.
    Expects data to be MNIST pytorch.
    """
    inputs = data.test_data[id]
    targets = eye(data.test_labels.shape[1], device=inputs.device)[target_label]

    return inputs, targets

def model_prediction(model, inputs):
    """
    Make a prediction for model given inputs.
    Returns: raw output, predicted class and raw output as string.
    """
    squeeze = len(inputs.shape) < 4
    if squeeze:
        inputs = inputs.unsqueeze(0)

    prob = model.predict(inputs)
    if squeeze:
        prob = prob[0]

    pred_class = argmax(prob, dim=-1).item()
    prob_str = space([round(x,1) for x in prob.tolist()], pred_class)

    return prob, pred_class, prob_str

def space(list, best):
    """Pretty print a list, with coloured highest number."""
    liststr = '['

    for i, number in enumerate(list):
        num = ''

        # Negatives need extra space.
        if number > 0:
            num += ' '
        # Nums betwen -10 and 10 need extra space
        if number < 10 and number > -10:
            num += ' '
        num += str(number)

        # Color if best
        if i == best:
            num = '\033[92m'+num+'\033[0m'
        liststr += num+', '

    return liststr[:-2] + ']'

# Example
# ae = load_AE('mnist_AE_weights', True)
