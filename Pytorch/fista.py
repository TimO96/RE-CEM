import sys
import torch
import numpy as np
import torchvision

def fista(label, beta, iter, max_iter, delta, slack, orig_img):
    cond1 = ((slack - orig_img) > beta).float32()
    cond2 = ((torch.abs(slack - orig_img)) <= beta).float32()
    cond3 = ((slack - orig_img) < -beta).float32()
    upper = torch.min((slack - beta), torch.tensor(0.5, dtype=torch.float32))
    lower = torch.max((slack + beta), torch.tensor(-0.5, dtype=torch.float32))

    delta_update = cond1 * upper + cond2 * orig_img + cond3 * lower

    cond4 = ((delta_update - orig_img) > 0).float32()
    cond5 = ((delta_update - orig_img) <= 0).float32()

    if label == "PP":
        delta_update = cond5 * delta_update + cond4 * orig_img
    elif label == "PN":
        delta_update = cond4 * delta_update + cond5 * orig_img

    zt = iter / (iter + torch.tensor(3, dtype=torch.float32)))
    slack_update = delta_update + zt * (delta_update - delta)

    cond6 = ((slack_update - orig_img) > 0).float32()
    cond7 = ((slack_update - orig_img) <= 0).float32()

    if label == "PP":
        slack_update = cond7 * slack_update + cond6 * orig_img
    elif label == "PN":
        slack_update = cond6 * slack_update + cond7 * orig_img

    return delta_update, slack_update
