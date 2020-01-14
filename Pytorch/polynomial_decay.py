def poly_lr_scheduler(optimizer, init_lr, step, lr_decay_step=1,
                      max_step=100, power=0.9):
    """
    Polynomial decay of learning rate
    Input:
        - optimizer     : initial optimizer
        - init_lr       : base learning rate
        - step          : current iteration
        - lr_decay_step : how frequently decay occurs, default is 1
        - max_step      : number of maximum iterations
        - power         : polymomial power
    Returns:
        - updated optimizer

    """
    if step % lr_decay_step or step > max_step:
        return optimizer

    lr = init_lr*(1 - step/max_step)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer