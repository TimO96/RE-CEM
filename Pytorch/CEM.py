## CEM.py -- Constrastive Explanation Method class in which adversarial
##           attacks are performed to analyze the target pertinent instance.
##
## (C) 2020 UvA FACT AI group

import sys
import tensorflow as tf
import torch
import numpy as np
import torchvision
import fista
import evaluation


class CEM:
    def __init__(self, model, mode, autoencoder, batch_size,  learning_rate_init,
                 c_init, c_steps, max_iterations, kappa, beta, gamma):
        """
        Constrastive Explanation Method (CEM) class initialization.
        Moreover, CEM is used to perform adversarial attacks with the autoencoder.
        Input:
            - model              : Pytorch CNN prediction model
            - mode               : perform either PN or PP analysis
            - autoencoder        : autoencoder model for the adversarial attacks
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

        # Initialize the shape for the variables used for pertinent analysis
        shape = (batch_size, \
                 model.image_size, model.image_size,  model.num_channels)

        # Define model variables
        self.model = model
        self.mode = mode
        self.autoencoder = autoencoder
        self.batch_size = batch_size
        self.lr_init = learning_rate_init
        self.c_init = c_init
        self.c_steps = c_steps
        self.max_iterations = max_iterations
        self.kappa = kappa
        self.beta = beta
        self.gamma = gamma

        # Define FISTA optimization variables
        self.fista_c = torch.zeros((batch_size,), dtype=tf.float32, requires_grad=True)
        self.adv_img_slack = torch.zeros((shape), dtype=torch.float32, requires_grad=True)
        self.optimizer = torch.optim.SGD(params=self.adv_img_slack, learning_rate = self.lr_init)

        # and here's what we use to assign them
        #self.assign_orig_img = torch.empty(shape, dtype=torch.float32)
        #self.assign_adv_img = torch.empty(shape, dtype=torch.float32)
        #self.assign_adv_img_s = torch.empty(shape, dtype=torch.float32)
        #self.assign_target_lab = torch.empty((batch_size, nun_classes), dtype=torch.float32)
        #self.assign_const = torch.empty(batch_size, dtype=torch.float32) ## ???? origineel


    def attack(self, imgs, labs):

        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                # x[y] -= self.kappa if self.PP else -self.kappa
                if self.mode == "PP":
                    x[y] -= self.kappa
                elif self.mode == "PN":
                    x[y] += self.kappa
                x = np.argmax(x)
            if self.mode == "PP":
                return x==y
            else:
                return x!=y

        batch_size = self.batch_size

        # set the lower and upper bounds accordingly
        Const_LB = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.init_const
        Const_UB = np.ones(batch_size) * 1e10
        # the best l2, score, and image attack
        overall_best_dist = [1e10] * batch_size
        overall_best_attack = [np.zeros(imgs[0].shape)] * batch_size

        for c_steps_idx in range(self.c_steps):
            # completely reset adam's internal state.
            img_batch = imgs[:batch_size]
            label_batch = labs[:batch_size]

            current_step_best_dist = [1e10] * batch_size
            current_step_best_score = [-1] * batch_size

            # set the variables so that we don't have to send them over again
            orig_img = img_batch
            target_lab = label_batch
            const = CONST
            adv_img = img_batch
            self.adv_img_slack = img_batch

            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                adv_img, self.adv_img_slack = fista.fista(self.mode, beta, iteration, adv_img, self.adv_img_slack, orig_img)
                self.optimizer.zero_grad()
                self.optimizer = poly_lr_scheduler(self.optimizer, self.lr_init, iteration)
                loss, loss_EN, OutputScore = evaluation.loss(adv_img-orig_img, orig_img, target_lab, kappa, AE, const, beta)
                loss.backward()
                self.optimizer.step()


                #Loss_Overall, Loss_EN, OutputScore, adv_img = self.sess.run([self.Loss_Overall, self.EN_dist, self.ImgToEnforceLabel_Score, self.adv_img])
                #Loss_Attack, Loss_L2Dist, Loss_L1Dist, Loss_AE_Dist = self.sess.run([self.Loss_Attack, self.Loss_L2Dist, self.Loss_L1Dist, self.Loss_AE_Dist])
                #target_lab_score, max_nontarget_lab_score_s = self.sess.run([self.target_lab_score, self.max_nontarget_lab_score])

                if iteration%(self.MAX_ITERATIONS//10) == 0:
                    print("iter:{} const:{}". format(iteration, CONST))
                    print("Loss_Overall:{:.4f}". format(loss))
                    #print("Loss_L2Dist:{:.4f}, Loss_L1Dist:{:.4f}, AE_loss:{}". format(Loss_L2Dist, Loss_L1Dist, Loss_AE_Dist))
                    #print("target_lab_score:{:.4f}, max_nontarget_lab_score:{:.4f}". format(target_lab_score[0], max_nontarget_lab_score_s[0]))
                    #print("")
                    sys.stdout.flush()

                for batch_idx,(the_dist, the_score, the_adv_img) in enumerate(zip(Loss_EN, OutputScore, adv_img)):
                    if the_dist < current_step_best_dist[batch_idx] and compare(the_score, np.argmax(label_batch[batch_idx])):
                        current_step_best_dist[batch_idx] = the_dist
                        current_step_best_score[batch_idx] = np.argmax(the_score)
                    if the_dist < overall_best_dist[batch_idx] and compare(the_score, np.argmax(label_batch[batch_idx])):
                        overall_best_dist[batch_idx] = the_dist
                        overall_best_attack[batch_idx] = the_adv_img

            # adjust the constant as needed
            for batch_idx in range(batch_size):
                if compare(current_step_best_score[batch_idx], np.argmax(label_batch[batch_idx])) and current_step_best_score[batch_idx] != -1:
                    # success, divide const by two
                    Const_UB[batch_idx] = min(Const_UB[batch_idx],CONST[batch_idx])
                    if Const_UB[batch_idx] < 1e9:
                        CONST[batch_idx] = (Const_LB[batch_idx] + Const_UB[batch_idx])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    Const_LB[batch_idx] = max(Const_LB[batch_idx],CONST[batch_idx])
                    if Const_UB[batch_idx] < 1e9:
                        CONST[batch_idx] = (Const_LB[batch_idx] + Const_UB[batch_idx])/2
                    else:
                        CONST[batch_idx] *= 10

        # return the best solution found
        overall_best_attack = overall_best_attack[0]
        return overall_best_attack.reshape((1,) + overall_best_attack.shape)
