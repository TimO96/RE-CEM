# methods.py -- Calculate objective function which combines the loss of the
#               autoencoder with a elastic net regularizer.
#               Also implements the Fast Iterative
#                Shrinkage-Thresholding Algorithm.

# (C) 2020 Changes by UvA FACT AI group [Pytorch conversion]

# Based on:
# Copyright (C) 2018, IBM Corp
#                     Chun-Chen Tu <timtu@umich.edu>
#                     PaiShun Ting <paishun@umich.edu>
#                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>

from torch import tensor, sum, abs, max, min


def eval_loss(model, mode, orig_img, adv, lab, autoencoder, c_start, kappa,
              gamma, beta, to_optimize=True):
    """
    Compute the loss function component for the network to find either
    pertinent positives (PP) or pertinent negatives (PN).
    Input:
        - model         : nn model
        - mode          : perform either PN or PP analysis
        - orig_img      : image from dataset
        - adv           : adversarial image
        - lab           : label of the to be predicted target class
        - autoecoder    : autoencoder model for the adversarial attacks
        - c_start       : regularization coefficient (hyperparameter)
        - kappa         : confidence parameter to measure the distance
                          between target class and other classes
        - gamma         : regularization weight for autoencoder loss function
        - beta          : regularization weight for the L1 loss term
        - to_optimize   : boolean which determines the option to optimize
    Returns:
        - computed loss between the most probable class and the most probable
          class given the pertubation (delta)
    """

    # Compute delta.
    delta = orig_img - adv

    # Distance to the input data.
    l2_loss, l1_loss = sum(delta**2), sum(abs(delta))
    elastic_dist = l2_loss + l1_loss * beta

    # Compute the total loss for the adversarial attack.
    nn_input = delta if (mode == "PP") else adv

    # Prediction before softmax of the model.
    pred = model.predict(nn_input.unsqueeze(0))[0]

    # Compute g(delta) which is the loss without the regularizers.
    loss_attack, lab_score, nonlab_score = loss_function(mode, pred, lab,
                                                         kappa)

    # Scale the current current c parameter with loss function f and sum to
    # retrieve a scalar.
    c_loss_attack = sum(c_start * loss_attack)

    # Based on the mode compute the last term of the objective function which
    # is the L2 reconstruction error of the autoencoder.
    ae_loss = gamma
    if gamma:
        ae_loss *= sum((autoencoder(nn_input.unsqueeze(0))[0] - nn_input)**2)

    # Determine whether the L1 loss term should be added when FISTA is not
    # optimized.
    loss = c_loss_attack + ae_loss + l2_loss
    if not to_optimize:
        loss += l1_loss * beta

    return loss, elastic_dist, pred, c_loss_attack, l2_loss, l1_loss, \
        lab_score, nonlab_score


def loss_function(mode, pred, target_lab, kappa):
    """
    Compute the loss function component for the network to find either
    pertinent positives (PP) or pertinent negatives (PN).
    Input:
        - mode          : perform either PN or PP analysis
        - pred          : prediction of model
        - target_lab    : label of the to be predicted target class
        - kappa         : confidence parameter to measure the distance
                          between target class and other classes
    Returns:
        - computed loss between the most probable class and the most probable
          class given the pertubation (delta) without regularizers.
    """

    # Compute the probability of the label class versus the other classes and
    # find the maximum in order to minimize the loss.
    lab_score = sum((target_lab) * pred)
    max_nonlab_score = max(pred[(1-target_lab).bool()])

    # Dependent on the mode subtract the score.
    if mode == "PP":
        f_loss = max_nonlab_score - lab_score
    elif mode == "PN":
        f_loss = lab_score - max_nonlab_score

    # Threshold the loss function f with a confidence parameter kappa.
    loss_attack = max(tensor([0.], device=pred.device), kappa + f_loss)

    return loss_attack, lab_score, max_nonlab_score


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

    half = tensor(0.5).to(z.device)

    # Apply FISTA conditions.
    delta_update = (z > beta) * min((slack - beta), half) + \
                   (abs(z) <= beta) * orig_img + \
                   (z < -beta) * max((slack + beta), -half)

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
    return (z > 0) * variable + (z <= 0) * orig_img
