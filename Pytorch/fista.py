## fista.py -- fast iterative shrinkage thresholding algorithm
##
## (C) 2020 UvA FACT AI group

import torch
from torch import tensor

def fista(label, beta, step, delta, slack, orig_img):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm.
    Input:
        - label    : label for orig_img
        - beta     : regularization coefficient (hyperparameter)
        - step     : k
        - delta    : last perturbation
        - slack    : last slack vector
        - orig_img : image from dataset
    Returns:
        - delta_update : new perturbation
        - slack_update : new slack vector
    """
    # Delta Update.
    z = slack - orig_img
    HALF = tensor(0.5).to(z.device)
    delta_update = (z > beta) * torch.min((slack - beta), HALF) + \
                   (torch.abs(z) <= beta) * orig_img + \
                   (z < -beta) * torch.max((slack + beta), -HALF)

    print(False in (delta_update.view(28*28) == orig_img.view(28*28)))
    delta_update = update(delta_update, orig_img, label)

    # Slack update.
    zt = step / (step + torch.tensor(3))
    slack_update = delta_update + zt * (delta_update - delta)
    slack_update = update(slack_update, orig_img, label)

    return delta_update, slack_update

def update(variable, orig_img, mode):
    """Update a variable based on label and its difference with orig_img."""
    z = variable - orig_img
    # print(z.device, variable.device, orig_img.device)
    if mode == "PP":
        return (z <= 0) * variable + (z > 0) * orig_img
    elif mode == "PN":
        return (z > 0) * variable + (z <= 0) * orig_img

    # When label is unknown.
    return None
