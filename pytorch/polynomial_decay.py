## poly_lr_scheduler -- polynomial learning rate scheduler for the attack.

## (C) 2020 Changes by UvA FACT AI group [Pytorch conversion]

## Based on:
## Copyright (C) 2018, IBM Corp
##                     Chun-Chen Tu <timtu@umich.edu>
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>


def poly_lr_scheduler(optimizer, init_lr, step, end_learning_rate=0.0001,
                      lr_decay_step=1, max_step=100000, power=1):
    """
    Polynomial decay of learning rate.
    Input:
        - optimizer     : initial optimizer
        - init_lr       : initial learning rate
        - step          : current iteration
        - lr_decay_step : how frequently decay occurs, default is 1
        - max_step      : number of maximum iterations
        - power         : polymomial power
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
