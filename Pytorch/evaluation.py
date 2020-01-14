## evaluation.py -- fast iterative shrinkage thresholding algorithm
##
## (C) 2020 UvA FACT AI group

def loss(delta, orig_img, target_lab, kappa, gamma, AE, const, beta, to_optimize=True, mode="PN"):
    """
    Compute the loss function component for the network to find either
    pertinent positives (PN) or pertinent negatives (PN).
    Input:
        - delta         : last perturbation
        - orig_img      : image from dataset
        - target_lab    : label of the to be predicted target class
        - kappa         : confidence parameter
        - gamma         : constant
        - AE            : autoencoder module
        - const         : regularization coefficient (hyperparameter)
        - beta          : regularization coefficient (hyperparameter)
        - to_optimize   : boolean which determines the option to optimize
        - mode          : find either PP or PN
    Returns:
        - computed loss between the most probable class and the most probable
          class given the pertubation (delta)
    """

    # Distance to the input data.
    L2_dist = torch.sum(delta**2, (1,2,3))
    L1_dist = torch.sum(torch.abs(delta), (1,2,3))
    EN_dist = L2_dist + L1_dist * beta

    loss_attack, pred = loss_function(delta, orig_img, target_lab, kappa, mode=mode)

    # Sum up the losses.
    loss_L1_dist = torch.sum(L1_dist)
    loss_L2_dist = torch.sum(L2_dist)
    loss_attack = torch.sum(const * loss_attack)

    if mode == "PP":
        loss_AE_dist   = gamma * (torch.norm(AE(delta) - delta)**2)
    elif mode == "PN":
        loss_AE_dist   = gamma * (torch.norm(AE(delta + orig_img) - delta + orig_img)**2)

    if to_optimize:
        loss = loss_attack + loss_L2_dist + loss_AE_dist
    else:
        loss = loss_attack + loss_L2_dist + loss_AE_dist + loss_L1_dist * beta

    return loss, EN_dist, pred

def loss_function(delta, orig_img, target_lab, kappa, mode="PN"):
    """
    Compute the loss function component for the network to find either
    pertinent positives (PN) or pertinent negatives (PN).
    Input:
        - delta         : last perturbation
        - orig_img      : image from dataset
        - target_lab    : label of the to be predicted target class
        - kappa         : confidence parameter
        - mode          : find either PP or PN
    Returns:
        - computed loss between the most probable class and the most probable
          class given the pertubation (delta) without regularizers.
    """

    # Prediction BEFORE-SOFTMAX of the model.
    if mode == "PP":
        pred = model.predict(delta)
    elif mode == "PN":
        pred = model.predict(orig_img + delta)

    # Compute the probability of the label class versus the maximum other.
    target_lab_score = torch.sum((target_lab) * pred, dim=1)
    # Inflate the real label in one-hot vector target_lab to infinity such that
    # the best class from the other classes is predicted.
    max_nontarget_lab_score = torch.max((1-target_lab) * pred - (target_lab*float('inf')), dim=1)

    if mode == "PP":
        loss_attack = torch.max(0.0, max_nontarget_lab_score - target_lab_score + kappa)
    elif mode == "PN":
        loss_attack = torch.max(0.0, -max_nontarget_lab_score + target_lab_score + kappa)

    return loss_attack, pred
