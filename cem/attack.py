# attack.py -- Attack class in which adversarial attacks are performed to
#              analyze the target pertinent instance.

# (C) 2020 Changes by UvA FACT AI group [Pytorch conversion]

# Based on:
# Copyright (C) 2018, IBM Corp
#                     Chun-Chen Tu <timtu@umich.edu>
#                     PaiShun Ting <paishun@umich.edu>
#                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>

import sys

from torch import argmax, zeros, no_grad
from torch.optim import SGD

from .methods import fista, eval_loss
from .utils import poly_lr_scheduler


class Attack:
    def __init__(self, model, AE, lr_init, c_init, c_steps, max_iterations,
                 kappa, beta, gamma, report):
        """
        Constrastive Explanation Method (CEM) class initialization.
        Moreover, CEM is used to perform adversarial attacks with the
        autoencoder.
        Input:
            - model          : Pytorch NN prediction model
            - AE             : autoencoder model for the adversarial attacks
            - lr_init        : starting learning rate for the optimizer
            - c_init         : starting weight constant of the loss function
            - c_steps        : number of iterations in which the constant c is
                               adjusted
            - max_iterations : maximum number of iterations for the analysis
            - kappa          : confidence parameter to measure the distance
                               between target class and other classes
            - beta           : regularization weight for the L1 loss term
            - gamma          : regularization weight for autoencoder loss
                               function
            - report         : print iterations
        """

        # Define model variables.
        self.model = model
        self.AE = AE
        self.lr_init = lr_init
        self.c_init = c_init
        self.c_steps = c_steps
        self.max_iterations = max_iterations
        self.kappa = kappa
        self.beta = beta
        self.gamma = gamma
        self.report = report
        self.mode = None

    def attack(self, img, lab, mode):
        """Perform attack on img.

        Input:
        - img  : model input
        - lab  : label for the image
        - mode : perform either PN or PP analysis

        Returns:
        - overall_best_attack: perturbation
        """
        self.mode = mode

        def compare(score, target):
            """
            Compare score with target label given the mode, ensure distance
            of kappa between target and max_nontarget and ensure loss_attack
            is thereby zero.
            """
            # Convert score to single number.
            if not isinstance(score, (float, int)):
                score = score.clone()
                kappa = self.kappa if (self.mode == "PP") else -self.kappa
                score[target] -= kappa
                score = argmax(score)

            # Depending on the mode compare the score to the ground truth.
            if self.mode == "PP":
                return score == target

            return score != target

        # Push image data to the available device.
        dvc = img.device

        # Set the lower and upper bounds, best distance, and image attack.
        lower_bound, upper_bound = 0, 1e10
        overall_best_dist = 1e10
        overall_best_attack = zeros(img.shape).to(dvc)

        # Initialize starting constant which will be adjusted accordingly.
        c_start = self.c_init

        # Find the best regularization coeffcient c by iterating through a
        # given number of steps.
        for _ in range(self.c_steps):
            # Initialize best distance and score. To be overwritten.
            current_step_best_dist = 1e10
            current_step_best_score = -1

            # Set the image x_0 and x (x_0 + delta) and its slack type which
            # is to be optimized.
            orig_img = img.clone()
            adv_img = img.clone() + img.clone().normal_(0, 0.03)
            # adv_img = torch.nn.init.kaiming_normal_(img.clone())
            # adv_img = img.clone().fill_(0)
            adv_img_slack = (adv_img.clone()).requires_grad_(True)

            # Initialize optimizer.
            optimizer = SGD(params=[adv_img_slack], lr=self.lr_init)

            # Iterate given number of steps to find the best attack for each c.
            for iteration in range(self.max_iterations):
                # perform the attack
                optimizer.zero_grad()
                optimizer = poly_lr_scheduler(optimizer, self.lr_init,
                                              iteration, 0., 0.5,
                                              max_step=self.max_iterations)

                # Compute the criterion which is used to optimize.
                loss, _, _, _, _, _, _, _ = eval_loss(self.model,
                                                      self.mode,
                                                      orig_img,
                                                      adv_img_slack,
                                                      lab,
                                                      self.AE,
                                                      c_start,
                                                      self.kappa,
                                                      self.gamma,
                                                      self.beta)

                # Apply gradients and update weights.
                loss.backward()
                optimizer.step()

                # Apply FISTA to update the to be optimized adversarial image.
                # This operation should not explicitly change the weights.
                with no_grad():
                    adv_img, adv_img_slack_update = fista(self.mode,
                                                          self.beta,
                                                          iteration+1,
                                                          adv_img,
                                                          adv_img_slack,
                                                          orig_img)
                    adv_img_slack.data = adv_img_slack_update.data

                # Estimate the loss after the FISTA update without optimizing.
                loss_no_opt, en_loss, pred, loss_attack, l2_loss, l1_loss,\
                lab_score, nonlab_score = eval_loss(self.model, self.mode,
                                                    orig_img, adv_img, lab,
                                                    self.AE, c_start,
                                                    self.kappa, self.gamma,
                                                    self.beta,
                                                    to_optimize=False)

                if iteration % (self.max_iterations//10) == 0 and self.report:
                    print(f"iter: {iteration} const: {c_start}")
                    print("Loss_Overall:{:.3f}, Loss_Elastic:{:.3f}"
                          .format(loss_no_opt, en_loss))
                    print("Loss_attack:{:.3f}, Loss_L2:{:.3f}, Loss_L1:{:.3f}"
                          .format(loss_attack, l2_loss, l1_loss))
                    print("lab_score:{:.3f}, max_nontarget_lab_score:{:.3f}"
                          .format(lab_score.item(), nonlab_score))
                    print("")
                    sys.stdout.flush()

                # Compare the score and ground truth and check the if the
                # distance obtained is higher than the best distance for the
                # current c and for the overall best c.
                comp = compare(pred, argmax(lab))
                if en_loss < current_step_best_dist and comp:
                    current_step_best_dist = en_loss
                    current_step_best_score = argmax(pred).item()
                if en_loss < overall_best_dist and comp:
                    overall_best_dist = en_loss
                    overall_best_attack = adv_img

            # Adjust the lower and upper bound based on the previously achieved
            # score of a c.
            if compare(current_step_best_score, argmax(lab)) and \
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
        return overall_best_attack
