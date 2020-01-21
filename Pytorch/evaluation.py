## evaluation.py -- objective function which combines the loss of the
##                  autoencoder with a elastic net regularizer.
##
## (C) 2020 UvA FACT AI group

import torch

def loss(model, mode, orig_img, adv, target_lab, AE, c_start, kappa,
         gamma, beta, to_optimize=True):
    """
    Compute the loss function component for the network to find either
    pertinent positives (PN) or pertinent negatives (PN).
    Input:
        - model         : nn model
        - mode          : perform either PN or PP analysis
        - orig_img      : image from dataset
        - delta         : last perturbation
        - target_lab    : label of the to be predicted target class
        - AE            : autoencoder model for the adversarial attacks
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

    delta = orig_img - adv

    # Distance to the input data.
    L2_dist = torch.sum(delta**2)
    L1_dist = torch.sum(torch.abs(delta))
    elastic_dist = L2_dist + L1_dist * beta

    # Calculate the total loss for the adversarial attack.
    loss_attack, pred, target_score, nontarget_score = loss_function(model, mode, adv, delta, target_lab, kappa)

    # Sum up the losses.
    loss_L1_dist = torch.sum(L1_dist)
    loss_L2_dist = torch.sum(L2_dist)

    loss_attack = torch.sum(c_start * loss_attack)

    # Based on the mode compute the last term of the objective function which
    # is the L2 reconstruction error of the autoencoder.
    loss_AE_dist = gamma
    if gamma:
        if mode == "PP":
            loss_AE_dist *= torch.norm(AE(delta.unsqueeze(0))[0] - delta)**2
        elif mode == "PN":
            loss_AE_dist *= (torch.norm(AE((delta + orig_img).unsqueeze(0))[0] - delta + \
                                    orig_img)**2)

    # Determine whether the L1 loss term should be added when FISTA is not
    # optimized.
    if to_optimize:
        loss = loss_attack + loss_L2_dist + loss_AE_dist
    else:
        loss = loss_attack + loss_L2_dist + loss_AE_dist + loss_L1_dist * beta

    return loss, elastic_dist, pred, loss_attack, loss_L2_dist, loss_L1_dist, target_score, nontarget_score

def loss_function(model, mode, adv, delta, target_lab, kappa):
    """
    Compute the loss function component for the network to find either
    pertinent positives (PN) or pertinent negatives (PN).
    Input:
        - model         : nn model
        - mode          : perform either PN or PP analysis
        - orig_img      : image from dataset
        - delta         : last perturbation
        - target_lab    : label of the to be predicted target class
        - kappa         : confidence parameter to measure the distance
                          between target class and other classes
    Returns:
        - computed loss between the most probable class and the most probable
          class given the pertubation (delta) without regularizers.
    """

    # Prediction before softmax of the model.
    if mode == "PP":
        pred = model.predict(delta.unsqueeze(0))[0]
    elif mode == "PN":
        pred = model.predict(adv.unsqueeze(0))[0]

    # print(pred)

    # Compute the probability of the label class versus the maximum others.
    target_lab_score = torch.sum((target_lab) * pred)
    # Inflate the real label in one-hot vector target_lab to infinity such that
    # the best class from the other classes is predicted.

    # print(target_lab)

    # max_nontarget_lab_score = torch.max((torch.ones(10)-target_lab) * pred - target_lab*10000
    max_nontarget_lab_score = torch.max(pred[(1-target_lab).bool()])

    zero = torch.tensor([0.], device=pred.device)
    if mode == "PP":
        loss_attack = torch.max(zero, max_nontarget_lab_score - \
                                target_lab_score + kappa)
    elif mode == "PN":
        # print(max_nontarget_lab_score)
        # print(type(max_nontarget_lab_score))
        loss_attack = torch.max(zero, -max_nontarget_lab_score + \
                                target_lab_score + kappa)

    return loss_attack, pred, target_lab_score, max_nontarget_lab_score
