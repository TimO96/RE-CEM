## fista.py -- Fast Iterative Shrinkage Thresholding Algorithm

## (C) 2020 Changes by UvA FACT AI group [Pytorch conversion]

## Based on:
## Copyright (C) 2018, IBM Corp
##                     Chun-Chen Tu <timtu@umich.edu>
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>

from torch import max, min, abs

def fista(mode, beta, step, delta, slack, orig_img):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm.
    Input:
        - mode     : mode for analysis
        - beta     : regularization coefficient (hyperparameter)
        - step     : k
        - delta    : last perturbation
        - slack    : last slack vector
        - orig_img : image from dataset
    Returns:
        - delta_update : new perturbation
        - slack_update : new slack vector
    """

    # Delta update.
    z = slack - orig_img

    HALF = tensor(0.5).to(z.device)

    # Apply FISTA conditions.
    delta_update = (z > beta) * tmin((slack - beta), HALF) + \
                   (tabs(z) <= beta) * orig_img + \
                   (z < -beta) * tmax((slack + beta), -HALF)

    # Apply delta update (delta^(k+1)).
    delta_update = update(delta_update, orig_img, mode)

    # Apply slack update (y^(k+1)) for momentum acceleration.
    zt = step / (step + tensor(3))
    slack_update = delta_update + zt * (delta_update - delta)
    slack_update = update(slack_update, orig_img, mode)

    return delta_update, slack_update

def update(variable, orig_img, mode):
    """Update a variable based on mode and its difference with orig_img."""
    
    # Apply the shrinkage-thresholding update element-wise.
    z = variable - orig_img
    if mode == "PP":
        return (z <= 0) * variable + (z > 0) * orig_img
    elif mode == "PN":
        return (z > 0) * variable + (z <= 0) * orig_img
