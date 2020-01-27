# utils.py -- Initialization of autoencoder and various utility functions.

# (C) 2020 Changes by UvA FACT AI group [Pytorch conversion]

# Based on:
# Copyright (C) 2018, IBM Corp
#                     Chun-Chen Tu <timtu@umich.edu>
#                     PaiShun Ting <paishun@umich.edu>
#                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>

import numpy as np

from torch import save, eye, argmax
from PIL import Image


def poly_lr_scheduler(optimizer, init_lr, step, end_learning_rate=0.0001,
                      lr_decay_step=1, max_step=100000, power=1):
    """
    Polynomial decay of learning rate.
    Input:
        - optimizer         : initial optimizer
        - init_lr           : initial learning rate
        - step              : current iteration
        - end_learning_rate : terminal learning rate
        - lr_decay_step     : how frequently decay occurs, default is 1
        - max_step          : number of maximum iterations
        - power             : polymomial power
    Returns:
        - updated optimizer
    """

    # Do not perform the scheduler if following conditions apply.
    if step % lr_decay_step or step > max_step:
        return optimizer

    # Apply the polymomial decay scheduler on the learning rate.
    lr = (init_lr - end_learning_rate)*(1 - step/max_step)**power + \
        end_learning_rate

    # Adjust the learning rate of the optimizer.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def save_img(img, name="output", channel=None, mode_img=None,
             save_tensor=False, thres=0, intensity=None):
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
            overlay = nfig.transpose(1, 2, 0)
        else:
            nfig[channel] = fig
            fig = nfig.transpose(1, 2, 0)

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
    targets = eye(data.test_labels.shape[1],
                  device=inputs.device)[target_label]

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
    prob_str = space([round(x, 1) for x in prob.tolist()], pred_class)

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
