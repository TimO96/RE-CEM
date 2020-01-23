## CEM.py -- Constrastive Explanation Method class in which adversarial
##           attacks are performed to analyze the target pertinent instance.

## (C) 2020 Changes by UvA FACT AI group [Pytorch conversion]

## Based on:
## Copyright (C) 2018, IBM Corp
##                     Chun-Chen Tu <timtu@umich.edu>
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>

import sys
from torch import argmax, zeros, no_grad, tensor
from torch import max as tmax
from torch import min as tmin
from torch import sum as tsum
from torch import norm as tnorm
from torch import abs as tabs
from torch.optim import SGD
import fista
import evaluation
from polynomial_decay import poly_lr_scheduler


class CEM:
    def __init__(self, model, mode, AE, lr_init, c_init, c_steps,
                 max_iterations, kappa, beta, gamma, report):
        """
        Constrastive Explanation Method (CEM) class initialization.
        Moreover, CEM is used to perform adversarial attacks with the
        autoencoder.
        Input:
            - model              : Pytorch NN prediction model
            - mode               : perform either PN or PP analysis
            - AE                 : autoencoder model for the adversarial attacks
            - lr_init            : starting learning rate for the optimizer
            - c_init             : starting weight constant of the loss function
            - c_steps            : number of iterations in which the constant c
                                   is adjusted
            - max_iterations     : maximum number of iterations for the analysis
            - kappa              : confidence parameter to measure the distance
                                   between target class and other classes
            - beta               : regularization weight for the L1 loss term
            - gamma              : regularization weight for autoencoder loss
                                   function
            - report             : print iterations
        """

        # Define model variables.
        self.model = model
        self.mode = mode
        self.AE = AE
        self.lr_init = lr_init
        self.c_init = c_init
        self.c_steps = c_steps
        self.max_iterations = max_iterations
        self.kappa = kappa
        self.beta = beta
        self.gamma = gamma
        self.report = report

    def attack(self, imgs, labs):
        """Perform attack on imgs."""

        def compare(score, target):
            """Compare score with target label given the mode."""

            # Convert score to single number.
            if not isinstance(score, (float, int)):
                score = score.clone()
                kappa = self.kappa if (self.mode == "PP") else -self.kappa
                score[target] -= kappa
                score = argmax(score)

            # Depending on the mode compare the score to the ground truth.
            if self.mode == "PP":
                return score == target
            elif self.mode == "PN":
                return score != target

        # Push image data to the available device.
        dvc = imgs.device

        # Set the lower and upper bounds, best distance, and image attack.
        lower_bound, upper_bound = 0, 1e10
        overall_best_dist = 1e10
        overall_best_attack = zeros(imgs.shape).to(dvc)

        # Initialize starting constant which will be adjusted accordingly.
        c_start = self.c_init

        # Find the best regularization coeffcient c by iterating through a
        # given number of steps.
        for c_steps_idx in range(self.c_steps):
            # Initialize best distance and score. To be overwritten.
            current_step_best_dist = 1e10
            current_step_best_score = -1

            # Set the image x_0 and x (x_0 + delta) and its slack type which
            # is to be optimized.
            x = imgs.clone()
            delta = imgs.clone().fill_(0)
            y = delta.clone().requires_grad_(True)
            t = labs.clone()
            # adv_img_slack = imgs.clone().fill_(0).requires_grad_(True)
            # adv_img = imgs.clone()
            # adv_img_slack = imgs.clone().requires_grad_(True)

            # Initialize optimizer.
            optimizer = SGD(params=[y], lr=self.lr_init)

            def f(mode, x, delta, target, kappa):
                if mode == "PP":
                    pred = self.model.predict(delta.unsqueeze(0))[0]
                    lab = tsum(target * pred)
                    nonlab = tmax(pred[(1-target).bool()])

                    # print(lab, nonlab)
                    # print(nonlab-lab)
                    # print(-kappa)

                    return tmax((nonlab - lab)+kappa, tensor(0.).to(dvc)), pred
                elif mode == "PN":
                    pred = self.model.predict((x+delta).unsqueeze(0))[0]
                    lab = tsum(target * pred)
                    nonlab = tmax(pred[(1-target).bool()])
                    return tmax((lab - nonlab)+kappa, tensor(0.).to(dvc)), pred

            def s(z, beta):
                HALF = tensor(0.5).to(z.device)

                return (z > beta) * tmin((z-beta), HALF) + \
                       (tabs(z) <= beta) * 0 + \
                       (z < -beta) * tmax((z+beta), -HALF)
                #
                #
                #
                # # Apply FISTA conditions.
                # delta_update = (z > beta) * min((slack - beta), HALF) + \
                #                (abs(z) <= beta) * orig_img + \
                #                (z < -beta) * max((slack + beta), -HALF)

            def project(mode, to_project, x):
                z = to_project - x
                if mode == "PP":
                    return (z <= 0) * to_project + (z > 0) * x
                elif mode == "PN":
                    return (z > 0) * to_project + (z <= 0) * x


            # Iterate a given number of steps to find the best attack for each c
            for iteration in range(self.max_iterations):
                # perform the attack
                optimizer.zero_grad()
                optimizer = poly_lr_scheduler(optimizer, self.lr_init,
                                              iteration, power=0.5,
                                              max_step=self.max_iterations)

                f_lss, pred = f(self.mode, x, y, t, self.kappa)
                g = c_start * f_lss + tsum(y**2)
                g.backward()
                optimizer.step()
                # ny = optimizer.param_groups[0]['params'][0].detach()

                to_project = s(y, self.beta)
                new_delta = project(self.mode, to_project, x)

                scale = iteration / (iteration + 3)
                to_project2 = new_delta + scale * (new_delta - delta)
                new_y = project(self.mode, to_project2, x)
                # .requires_grad_(True)

                delta = new_delta
                # optimizer.param_groups[0]['params'][0] = new_y
                y.data = new_y.data

                loss_EN = self.beta * tsum(tabs(delta)) +tsum(delta**2)


                # Compute the criterion which is used to optimize.
                # loss, _, _, _, _, _, _, _ = evaluation.loss(self.model,
                    # self.mode, orig_img, adv_img_slack, labs, self.AE, c_start,
                    # self.kappa, self.gamma, self.beta)

                # Apply gradients and update weights.


                # # Apply FISTA to update the to be optimized adversarial image.
                # # This operation should not explicitly change the weights.
                # with no_grad():
                #     adv_img, adv_img_slack_update = fista.fista(self.mode,
                #         self.beta, iteration, adv_img, adv_img_slack, orig_img)
                #     adv_img_slack.data = adv_img_slack_update.data
                #
                # # Estimate the losses after the FISTA update without optimizing.
                # loss_no_opt, loss_EN, pred, loss_attack, loss_L2_dist, \
                # loss_L1_dist, target_score, nontarget_score = evaluation.loss(
                #      self.model, self.mode, orig_img, adv_img, labs,
                #      self.AE, c_start, self.kappa, self.gamma, self.beta,
                #      to_optimize=False)

                if iteration%(self.max_iterations//10) == 0 and self.report:
                    print(f"iter: {iteration} const: {c_start}")
                    print(f"lss_k: {loss_EN}")
                    print(f"lss_attack {g}")
                    # print("Loss_Overall:{:.3f}, Loss_Elastic:{:.3f}"\
                          # .format(loss_no_opt, loss_EN))
                    # print("Loss_attack:{:.3f}, Loss_L2:{:.3f}, Loss_L1:{:.3f}"\
                          # .format(loss_attack, loss_L2_dist, loss_L1_dist))
                    # print("labs_score:{:.3f}, max_nontarget_lab_score:{:.3f}"\
                          # .format(target_score.item(), nontarget_score))
                    print("")
                    sys.stdout.flush()

                # Compare the score and ground truth and check the if the
                # distance obtained is higher than the best distance for the
                # current c and for the overall best c.
                comp = compare(pred, argmax(t))
                if loss_EN < current_step_best_dist and comp:
                    current_step_best_dist = loss_EN
                    current_step_best_score = argmax(pred).item()
                if loss_EN < overall_best_dist and comp:
                    overall_best_dist = loss_EN
                    overall_best_attack = delta


            # Adjust the lower and upper bound based on the previously achieved
            # score of a c.
            if compare(current_step_best_score, argmax(t)) and \
               current_step_best_score != -1:
                # success, divide const by two
                upper_bound = min(upper_bound, c_start)
                if upper_bound < 1e9:
                    c_start = (lower_bound + upper_bound)/2
            # If no proper solution is found: upscale the constant c value with
            # a factor of 10. Else interpolate between the boundary values.
            else:
                lower_bound = max(lower_bound, c_start)
                if upper_bound < 1e9:
                    c_start = (lower_bound + upper_bound)/2
                else:
                    c_start *= 10


        # Return the overall best attack between the different c values.
        return overall_best_attack.detach()
