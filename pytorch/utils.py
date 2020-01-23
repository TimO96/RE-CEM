## utils.py -- Initialization of autoencoder and various utility functions.

## (C) 2020 Changes by UvA FACT AI group [Pytorch conversion]

## Based on:
## Copyright (C) 2018, IBM Corp
##                     Chun-Chen Tu <timtu@umich.edu>
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>

import os
import h5py
import numpy as np

from torch.nn import Module, Conv2d, LeakyReLU, MaxPool2d, Upsample, Sequential
from torch import load, save, eye, from_numpy, argmax
from torchsummary import summary
from PIL import Image


class AE(Module):
    def __init__(self, restore=None):
        """
        Autoencoder architecture based on mnist_AE_1_decoder.json.
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


def h5_to_state_dict(h5_file, mapping):
    """Create Pytorch state_dict from h5 weight with mapping."""

    state_dict = {}
    with h5py.File(h5_file, 'r') as f:
        for h5, state in mapping.items():
            # Weights in tensorflow are transposed with respect to Pytorch.
            state_dict[state] = from_numpy(f[h5][:].T)

    return state_dict

def load_AE(codec_prefix, print_summary=False, dir="models/"):
    """Load autoencoder from json. Optionally print summary."""

    # Weights file.
    weight_file = dir + codec_prefix  + ".h5"
    if not os.path.isfile(weight_file):
        raise Exception(f"Decoder weight file {weight_file} not found.")

    # AE mapping from keras format to Pytorch.
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

    # Create autoencoder instance.
    ae = AE(h5_to_state_dict(weight_file, AE_map))

    # Print autoencoder structure.
    if print_summary:
        summary(ae, (ae.image_size, ae.image_size, ae.num_channels))

    return ae

def save_img(img, name="output", channel=None, mode_img=None, save_tensor=False,
             thres=10, intensity=None):
    """Save an MNIST image to location name, both as .pt and .png."""

    # Save image tensor.
    if save_tensor:
        save(img, name+'.pt')

    # Save image, invert MNIST read.
    fig = np.around((img.cpu().data.numpy() + 0.5) * 255)
    fig = fig.astype(np.uint8).squeeze()

    # Apply colors to indicate the PN and PP pixels.
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

    # Name = 'output'
    pic.save(name+'.png')

    return pic

def generate_data(data, id, target_label):
    """
    Return test data id and one hot target.
    Expects data to be MNIST Pytorch.
    """

    inputs = data.test_data[id]
    targets = eye(data.test_labels.shape[1], device=inputs.device)[target_label]

    return inputs, targets

def model_prediction(model, inputs):
    """
    Make a prediction for model given inputs.
    Returns: raw output, predicted class and raw output as string.
    """

    # Unsqueeze if input is still shaped for batched inputs.
    squeeze = len(inputs.shape) < 4
    if squeeze:
        inputs = inputs.unsqueeze(0)

    # Retrieve model outut predictions in probabilities without activations.
    prob = model.predict(inputs)
    if squeeze:
        prob = prob[0]

    # Retrieve predicted class.
    pred_class = argmax(prob, dim=-1).item()

    # Pretty print of the output prediction.
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
        # Numbers between -10 and 10 need extra space
        if number < 10 and number > -10:
            num += ' '
        num += str(number)

        # Highlight best scores in green color.
        if i == best:
            num = '\033[92m'+num+'\033[0m'
        liststr += num+', '

    return liststr[:-2] + ']'
