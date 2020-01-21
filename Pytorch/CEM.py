## CEM.py -- Constrastive Explanation Method class in which adversarial
##           attacks are performed to analyze the target pertinent instance.
##
## (C) 2020 UvA FACT AI group

import sys
import torch
import numpy as np
import fista
import evaluation
from polynomial_decay import poly_lr_scheduler
from torchvision import utils
from torch.autograd import Variable
from torch import nn
import ipdb


class CEM:
    def __init__(self, model, mode, AE, learning_rate_init, c_init, c_steps,
                 max_iterations, kappa, beta, gamma):
        """
        Constrastive Explanation Method (CEM) class initialization.
        Moreover, CEM is used to perform adversarial attacks with the autoencoder.
        Input:
            - model              : Pytorch CNN prediction model
            - mode               : perform either PN or PP analysis
            - AE                 : autoencoder model for the adversarial attacks
            - batch_size         : number of data instances to be analyzed
            - learning_rate_init : starting learning rate for the optimizer
            - c_init             : starting weight constant of the loss function
            - c_steps            : number of iterations in which the constant c
                                   is adjusted
            - max_iterations     : maximum number of iterations for the analysis
            - kappa              : confidence parameter to measure the distance
                                   between target class and other classes
            - beta               : regularization weight for the L1 loss term
            - gamma              : regularization weight for autoencoder loss
                                   function
        """
        # Define model variables
        self.model = model
        self.mode = mode
        self.AE = AE
        self.lr_init = learning_rate_init
        self.c_init = c_init
        self.c_steps = c_steps
        self.max_iterations = max_iterations
        self.kappa = kappa
        self.beta = beta
        self.gamma = gamma

    def attack(self, imgs, labs):
        """Perform attack on imgs."""

        def compare(score, target):
            """Compare score with target."""
            # Convert score to single number
            if not isinstance(score, (float, int)):
                score = score.clone()
                kappa = self.kappa if (self.mode == "PP") else -self.kappa
                score[target] -= kappa
                score = torch.argmax(score)

            if self.mode == "PP":
                return score == target
            elif self.mode == "PN":
                return score != target

        dvc = imgs.device

        # set the lower and upper bounds, best distance, and image attack
        lower_bound, c_start, upper_bound = 0, self.c_init, 1e10
        overall_best_dist = 1e10
        overall_best_attack = torch.zeros(imgs.shape).to(dvc)

        for c_steps_idx in range(self.c_steps):
            current_step_best_dist = 1e10
            current_step_best_score = -1

            # set the variables so that we don't have to send them over again
            orig_img = imgs.clone()
            adv_img = imgs.clone().fill_(0)
            adv_img_slack = imgs.clone().fill_(0).requires_grad_(True)

            optimizer = torch.optim.SGD(params=[adv_img_slack], lr=self.lr_init)

            for iteration in range(self.max_iterations):
                # perform the attack
                optimizer.zero_grad()
                optimizer = poly_lr_scheduler(optimizer, self.lr_init, iteration)

                # Optimize first part.
                loss, _, _, _, _, _, _, _ = evaluation.loss(self.model,
                    self.mode, orig_img, adv_img_slack, labs, self.AE, c_start,
                    self.kappa, self.gamma, self.beta)
                loss.backward()
                optimizer.step()

                # Optimize second part.
                with torch.no_grad():
                    adv_img, adv_img_slack_update = fista.fista(self.mode,
                        self.beta, iteration, adv_img, adv_img_slack, orig_img)
                adv_img_slack.data = adv_img_slack_update.data

                # Get losses.
                loss_no_opt, loss_EN, pred, loss_attack, loss_L2_dist, \
                loss_L1_dist, target_score, nontarget_score = evaluation.loss(
                     self.model, self.mode, orig_img, adv_img, labs,
                     self.AE, c_start, self.kappa, self.gamma, self.beta,
                     to_optimize=False)

                if iteration%(self.max_iterations//10) == 0:
                    print(f"iter: {iteration} const: {c_start}")
                    print("Loss_Overall:{:.3f}, Loss_Elastic:{:.3f}". format(loss_no_opt, loss_EN))
                    print("Loss_attack:{:.3f}, Loss_L2:{:.3f}, Loss_L1:{:.3f}". format(loss_attack, loss_L2_dist, loss_L1_dist))
                    print("labs_score:{:.3f}, max_nontarget_lab_score:{:.3f}". format(target_score.item(), nontarget_score))
                    print("")
                    sys.stdout.flush()

                # Update current & global.
                comp = compare(pred, torch.argmax(labs))
                if loss_EN < current_step_best_dist and comp:
                    current_step_best_dist = loss_EN
                    current_step_best_score = torch.argmax(pred).item()
                if loss_EN < overall_best_dist and comp:
                    overall_best_dist = loss_EN
                    overall_best_attack = adv_img

            # adjust the constant as needed
            if compare(current_step_best_score, torch.argmax(labs)) and \
               current_step_best_score != -1:
                # success, divide const by two
                upper_bound = min(upper_bound, c_start)
                if upper_bound < 1e9:
                    c_start = (lower_bound + upper_bound)/2
            else:
                # failure, either multiply by 10 if no solution found yet
                #          or do binary search with the known upper bound
                lower_bound = max(lower_bound, c_start)
                if upper_bound < 1e9:
                    c_start = (lower_bound + upper_bound)/2
                else:
                    c_start *= 10

        # return the best solution found
        return overall_best_attack
