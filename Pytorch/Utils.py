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


# from tensorflow.keras.models import Model, model_from_json, Sequential
# from torchvision.utils import save_image
import torch
# from PIL import Image
# import tensorflow as tf
import os
import numpy as np


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
    torch.save(img, name+'.pt')

    # Save image, invert MNIST read
    fig = np.around((img + 0.5) * 255)
    fig = fig.astype(np.uint8).squeeze()
    save_image(fig, name+'.png')

def generate_data(data, id, target_label):
    """
    Return test data id and one hot target.
    Expects data to be MNIST pytorch.
    """
    inputs = data.test_data[id]
    targets = torch.eye(data.test_labels.shape[1])[target_label])

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
