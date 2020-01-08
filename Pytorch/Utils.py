## Utils.py -- Some utility functions
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
from torch import save, eye, uint8

import os

class Print(Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

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

        model = [
            #
            # Encoder
            #
            Conv2d(self.num_channels, self.filter_size, self.kernel_size,
                   padding=self.padding),
            LeakyReLU(self.relu_slope),
            Conv2d(self.filter_size, self.filter_size, self.kernel_size,
                   padding=self.padding),
            LeakyReLU(self.relu_slope),
            MaxPool2d(self.pool_kernel_size),
            Conv2d(self.filter_size, self.num_channels, self.kernel_size,
                   padding=self.padding),
            #
            # Decoder
            #
            Conv2d(self.num_channels, self.filter_size, self.kernel_size,
                   padding=self.padding),
            LeakyReLU(self.relu_slope),
            Upsample(scale_factor=self.pool_kernel_size),
            # Upsample(size=self.pool_kernel_size),
            Conv2d(self.filter_size, self.filter_size, self.kernel_size,
                   padding=self.padding),
            LeakyReLU(self.relu_slope),
            Conv2d(self.filter_size, self.num_channels, self.kernel_size,
                   padding=self.padding),
        ]

        if restore:
            try:
                self.load_state_dict(restore)
            except:
                print("Error:", sys.exc_info()[0])
                print("Make sure restore is a torch.load(PATH) object")
                raise

        self.model = Sequential(*model)

    def predict(self, data):
        """Predict output for input data (batch, dim1, dim2, c)."""
        assert data[0].shape == (28, 28, 1), "Expected shape (b, 28, 28, 1)."

        # Reshape data, expect (batch, channel, dim1, dim2)
        pred = self.model(data.view(-1, 1, self.image_size, self.image_size))

        return pred.view(-1, self.image_size, self.image_size, 1)

m = AE()
print(m)
from setup_mnist import *
data = MNIST()
p = m.predict(data.test_data[0:10])
print(p.shape)

def load_AE(codec_prefix, print_summary=False, dir="models/AE_codec/"):
    """Load autoencoder from json, optionally print summary."""
    # Decoder files
    prefix = dir + codec_prefix + "_"
    decoder_model_file = prefix + "decoder.json"
    decoder_weight_file = prefix + "decoder.h5"

    if not os.path.isfile(decoder_model_file):
        raise Exception(f"Decoder model file {decoder_model_file} not found.")
    if not os.path.isfile(decoder_weight_file):
        raise Exception(f"Decoder weight file {decoder_weight_file} not found.")

    json_file = open(decoder_model_file, 'r')
    decoder = model_from_json(json_file.read(), custom_objects={"tf": tf})
    json_file.close()

    decoder.load_weights(decoder_weight_file)

    if print_summary:
        print("Decoder summaries")
        decoder.summary()

    return decoder

def save_img(img, name="output"):
    """Save an MNIST image to location name, both as .pt and .png."""
    # Save tensor
    save(img, name+'.pt')

    # Save image, invert MNIST read
    fig = ((img + 0.5) * 255).round()
    fig = fig.type(uint8).squeeze()
    save_image(fig, name+'.png')

def generate_data(data, id, target_label):
    """
    Return test data id and one hot target.
    Expects data to be MNIST pytorch.
    """
    inputs = data.test_data[id]
    targets = eye(data.test_labels.shape[1])[target_label]

    return inputs, target

def model_prediction(model, inputs):
    """
    Make a prediction for model given inputs.
    Returns: raw output, predicted class and raw output as string.
    """
    prob = model.predict(inputs)
    predicted_class = torch.argmax(prob, dim=-1)
    prob_str = np.array2string(prob.cpu().data.numpy()).replace('\n','')

    return prob, predicted_class, prob_str
