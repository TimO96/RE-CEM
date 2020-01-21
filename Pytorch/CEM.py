## CEM.py -- Constrastive Explanation Method class in which adversarial
##           attacks are performed to analyze the target pertinent instance.
##
## (C) 2020 UvA FACT AI group

import sys
import torch
import numpy as np
#import torchvision
import fista
import evaluation
from polynomial_decay import poly_lr_scheduler
from torchvision import utils
from torch.autograd import Variable
from torch import nn


class CEM:
    def __init__(self, model, mode, AE, batch_size, learning_rate_init,
                 c_init, c_steps, max_iterations, kappa, beta, gamma):
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
        self.batch_size = batch_size
        self.lr_init = learning_rate_init
        self.c_init = c_init
        self.c_steps = c_steps
        self.max_iterations = max_iterations
        self.kappa = kappa
        self.beta = beta
        self.gamma = gamma

    def attack(self, imgs, labs):
        """Perform attack on imgs."""
        dvc = imgs.device

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

        # batch_size = self.batch_size

        # set the lower and upper bounds accordingly
        # lower_bound = torch.zeros(batch_size).to(dvc)
        # c_start = torch.ones(batch_size).to(dvc) * self.c_init
        # upper_bound = torch.ones(batch_size).to(dvc) * 1e10

        lower_bound = 0
        c_start = self.c_init
        upper_bound = 1e10

        # the best l2, score, and image attack
        overall_best_dist = 1e10
        overall_best_attack = torch.zeros(imgs[0].shape).to(dvc)
        # img_batch = imgs[:batch_size]
        # labs = labs[:batch_size]
        #label_batch = labs[:batch_size]

        for c_steps_idx in range(self.c_steps):
            # completely reset adam's internal state.

            current_step_best_dist = 1e10
            current_step_best_score = -1

            # set the variables so that we don't have to send them over again
            orig_img = imgs.clone()
            adv_img = imgs.clone()
            adv_img_slack = imgs.clone().requires_grad_(True)

            optimizer = torch.optim.SGD(params=[adv_img_slack], lr=self.lr_init)
            #utils.save_image(orig_img.squeeze(), 'original_image.png')

            # q = adv_img_slack.clone()

            for iteration in range(self.max_iterations):
                # perform the attack
                optimizer.zero_grad()
                optimizer = poly_lr_scheduler(optimizer, self.lr_init, iteration)
                loss, _, _, _, _, _, _, _ = evaluation.loss(self.model,
                                                            self.mode,
                                                            orig_img,
                                                            adv_img_slack,
                                                            labs,
                                                            self.AE,
                                                            c_start,
                                                            self.kappa,
                                                            self.gamma,
                                                            self.beta)

                loss.backward()
                optimizer.step()

                # adv = optimizer.param_groups[0]['params'][0]
                # print(ad)
                # print(False in (q.view(28*28) == adv.view(28*28)))

                with torch.no_grad():
                    adv_img, adv_img_slack_update = fista.fista(self.mode,
                                                                self.beta,
                                                                iteration,
                                                                adv_img,
                                                                adv_img_slack,
                                                                orig_img)

                # adv_img_slack.data = adv_img_slack_update.data
                adv_img_slack.data = adv_img_slack_update.data

                loss_no_opt, loss_EN, pred, loss_attack, loss_L2_dist, \
                loss_L1_dist, target_score, nontarget_score = evaluation.loss(
                     self.model, self.mode, orig_img, adv_img, labs,
                     self.AE, c_start, self.kappa, self.gamma, self.beta,
                     to_optimize=False)


                if iteration%(self.max_iterations//10) == 0:
                    print(f"iter: {iteration} const: {c_start}")
                    # print(loss_no_opt, loss_EN)
                    print("Loss_Overall:{:.3f}, Loss_Elastic:{:.3f}". format(loss_no_opt, loss_EN))
                    print("Loss_attack:{:.3f}, Loss_L2:{:.3f}, Loss_L1:{:.3f}". format(loss_attack, loss_L2_dist, loss_L1_dist))
                    #utils.save_image(adv_img.detach().squeeze(), str(c_steps_idx) + '-' + str(iteration) + '-img.png')
                    #print("Loss_L2Dist:{:.4f}, Loss_L1Dist:{:.4f}, AE_loss:{}". format(Loss_L2Dist, Loss_L1Dist, Loss_AE_Dist))
                    print("labs_score:{:.3f}, max_nontarget_lab_score:{:.3f}". format(target_score.item(), nontarget_score))
                    print("")
                    sys.stdout.flush()

                # for batch_idx,(dist, score, the_adv_img) in enumerate(zip(loss_EN, pred, adv_img)):
                comp = compare(pred, torch.argmax(labs))

                #
                if loss_EN < current_step_best_dist and comp:
                    current_step_best_dist = loss_EN
                    current_step_best_score = torch.argmax(pred).item()

                #
                if loss_EN < overall_best_dist and comp:
                    overall_best_dist = loss_EN
                    overall_best_attack = adv_img

            # adjust the constant as needed
            # for batch_idx in range(batch_size):
            if compare(current_step_best_score, torch.argmax(labs)) and current_step_best_score != -1:
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
