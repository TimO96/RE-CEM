def loss(delta, orig_img, target_lab, kappa, AE, const, beta, to_optimize=True mode="PN"):

    # distance to the input data
    L2_dist = torch.sum(delta**2, (1,2,3))
    L1_dist = torch.sum(torch.abs(delta), (1,2,3))
    EN_dist = L2_dist + L1_dist * beta

    Loss_Attack = loss_function(delta, orig_img, target_lab, kappa, mode=mode)

    # sum up the losses
    Loss_L1Dist = torch.sum(L1_dist)
    Loss_L2Dist = torch.sum(L2_dist)
    Loss_Attack = torch.sum(const * Loss_Attack)

    if mode == "PP":
        Loss_AE_Dist   = gamma * (torch.norm(AE(delta) - delta)**2)
    elif mode == "PN":
        Loss_AE_Dist   = gamma * (torch.norm(AE(delta + orig_img) - delta + orig_img)**2)

    if to_optimize:
        return Loss_Attack + Loss_L2Dist + Loss_AE_Dist
    else:
        return Loss_Attack + Loss_L2Dist + Loss_AE_Dist + torch.mul(beta, Loss_L1Dist)

def loss_function(delta, orig_img, target_lab, kappa, mode="PN"):
    # prediction BEFORE-SOFTMAX of the model
    if mode == "PP":
        pred = model.predict(delta)
    elif mode == "PN":
        pred = model.predict(orig_img + delta)

    # compute the probability of the label class versus the maximum other
    target_lab_score = torch.sum((target_lab) * pred, 1)
    max_nontarget_lab_score = torch.max((1-target_lab) * pred - (target_lab*10000), dim=1)

    if mode == "PP":
        Loss_Attack = torch.max(0.0, max_nontarget_lab_score - target_lab_score + kappa)
    elif mode == "PN":
        Loss_Attack = torch.max(0.0, -max_nontarget_lab_score + target_lab_score + kappa)