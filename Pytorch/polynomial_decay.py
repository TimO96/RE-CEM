def poly_lr_scheduler(optimizer, init_lr, step, end_learning_rate=0.0001, lr_decay_step=1, max_step=100000, power=1):
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

    lr = (init_lr - end_learning_rate)*(1 - step/max_step)**power + end_learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer