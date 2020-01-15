## aen_attack.py -- attack a network optimizing elastic-net distance with an en decision rule
##                  when autoencoder loss is applied
##
## Copyright (C) 2018, IBM Corp
##                     Chun-Chen Tu <timtu@umich.edu>
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import sys
import tensorflow as tf
import torch
import numpy as np
import torchvision
import fista
import evaluation


class CEM:
    def __init__(self, sess, model, mode, AE, batch_size, kappa, init_learning_rate,
                 binary_search_steps, max_iterations, initial_const, beta, gamma):


        image_size, num_channels, nun_classes = model.image_size, model.num_channels, model.num_labels
        shape = (batch_size, image_size, image_size, num_channels)

        self.sess = sess
        self.INIT_LEARNING_RATE = init_learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.kappa = kappa
        self.init_const = initial_const
        self.batch_size = batch_size
        self.AE = AE
        self.mode = mode
        self.beta = beta
        self.gamma = gamma

        # these are variables to be more efficient in sending data to tf
        self.orig_img = torch.zeros((shape), dtype=torch.float32, requires_grad=True)
        #self.orig_img = Variable(torch.zepe=torch.float32))
        self.adv_img = torch.zeros((shape), dtype=torch.float32, requires_grad=True)
        self.adv_img_s = torch.zeros((shape), dtype=torch.float32, requires_grad=True)
        self.target_lab = torch.zeros((batch_size, nun_classes), dtype=torch.float32)
        self.const = torch.zeros((batch_size,), dtype=tf.float32, requires_grad=True)
        self.global_step = torch.tensor(0.0, requires_grad=False)

        # and here's what we use to assign them
        self.assign_orig_img = torch.empty(shape, dtype=torch.float32)
        self.assign_adv_img = torch.empty(shape, dtype=torch.float32)
        self.assign_adv_img_s = torch.empty(shape, dtype=torch.float32)
        self.assign_target_lab = torch.empty((batch_size, nun_classes), dtype=torch.float32)
        self.assign_const = torch.empty(batch_size, dtype=torch.float32) ## ???? origineel

        self.learning_rate = polynomial_decay(self.INIT_LEARNING_RATE, self.global_step, self.MAX_ITERATIONS, 0, power=0.5)
        optimizer = torch.optim.SGD(params=?, lr=self.learning_rate)

    def attack(self, imgs, labs):

        def compare(x,y):
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

        for binary_search_steps_idx in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            img_batch = imgs[:batch_size]
            label_batch = labs[:batch_size]

            current_step_best_dist = [1e10] * batch_size
            current_step_best_score = [-1] * batch_size

            # set the variables so that we don't have to send them over again
            assign_orig_img = img_batch,
            assign_target_lab = label_batch
            assign_const = CONST
            assign_adv_img = img_batch
            assign_adv_img_s = img_batch

            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                optimizer.zero_grad()
                orig_img = delta = slack = img_batch
                delta, slack = fista(self.mode, self.beta, iteration, delta, slack, orig_img):

                Loss_Overall, Loss_EN, OutputScore, adv_img = self.sess.run([self.Loss_Overall, self.EN_dist, self.ImgToEnforceLabel_Score, self.adv_img])
                Loss_Attack, Loss_L2Dist, Loss_L1Dist, Loss_AE_Dist = self.sess.run([self.Loss_Attack, self.Loss_L2Dist, self.Loss_L1Dist, self.Loss_AE_Dist])
                target_lab_score, max_nontarget_lab_score_s = self.sess.run([self.target_lab_score, self.max_nontarget_lab_score])

                if iteration%(self.MAX_ITERATIONS//10) == 0:
                    print("iter:{} const:{}". format(iteration, CONST))
                    print("Loss_Overall:{:.4f}, Loss_Attack:{:.4f}". format(Loss_Overall, Loss_Attack))
                    print("Loss_L2Dist:{:.4f}, Loss_L1Dist:{:.4f}, AE_loss:{}". format(Loss_L2Dist, Loss_L1Dist, Loss_AE_Dist))
                    print("target_lab_score:{:.4f}, max_nontarget_lab_score:{:.4f}". format(target_lab_score[0], max_nontarget_lab_score_s[0]))
                    print("")
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
        return overalla_best_attack.reshape((1,) + overall_best_attack.shape)
