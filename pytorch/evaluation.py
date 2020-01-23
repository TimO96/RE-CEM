## evaluation.py -- objective function which combines the loss of the
##                  autoencoder with a elastic net regularizer.
##
## (C) 2020 UvA FACT AI group

from torch import tensor, sum, abs, max


def loss(model, mode, orig_img, adv, lab, AE, c_start, kappa, gamma, beta,
         to_optimize=True):
    """
    Compute the loss function component for the network to find either
    pertinent positives (PP) or pertinent negatives (PN).
    Input:
        - model         : nn model
        - mode          : perform either PN or PP analysis
        - orig_img      : image from dataset
        - adv           : adversarial image
        - lab           : label of the to be predicted target class
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

    # Compute delta.
    delta = orig_img - adv

    # Distance to the input data.
    loss_L2, loss_L1 = sum(delta**2), sum(abs(delta))
    elastic_dist = loss_L2 + loss_L1 * beta

    # Compute the total loss for the adversarial attack.
    input = delta if (mode == "PP") else adv

    # Prediction before softmax of the model.
    pred = model.predict(input.unsqueeze(0))[0]

    # Compute g(delta) which is the loss without the regularizers.
    loss_attack, lab_score, nonlab_score = loss_function(mode, pred, lab, kappa,
                                                         c_start)

    # Scale the current current c parameter with loss function f and sum to
    # retrieve a scalar.
    c_loss_attack = sum(c_start * loss_attack)

    # Based on the mode compute the last term of the objective function which
    # is the L2 reconstruction error of the autoencoder.
    loss_AE = gamma
    if gamma:
        loss_AE *= sum((AE(input.unsqueeze(0))[0] - input)**2)

    # Determine whether the L1 loss term should be added when FISTA is not
    # optimized.
    loss = c_loss_attack + loss_AE + loss_L2
    if not to_optimize:
         loss += loss_L1 * beta

    return loss, elastic_dist, pred, c_loss_attack, loss_L2, loss_L1, lab_score, \
           nonlab_score

def loss_function(mode, pred, target_lab, kappa, c_start):
    """
    Compute the loss function component for the network to find either
    pertinent positives (PP) or pertinent negatives (PN).
    Input:
        - mode          : perform either PN or PP analysis
        - pred          : prediction of model
        - target_lab    : label of the to be predicted target class
        - kappa         : confidence parameter to measure the distance
                          between target class and other classes
        - c_start       : regularization coefficient (hyperparameter)
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
        f = max_nonlab_score - lab_score
    elif mode == "PN":
        f = lab_score - max_nonlab_score

    # Threshold the loss function f with a confidence parameter kappa.
    loss_attack = max(tensor([0.], device=pred.device), kappa + f)

    return loss_attack, lab_score, max_nonlab_score
