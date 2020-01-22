## main.py -- sample code to test attack procedure
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

## (C) 2020 Changes by UvA FACT AI group [Pytorch conversion]

import os
import sys
import random
import time
from setup_mnist import MNIST, MNISTModel
import torch
from torch import cuda, manual_seed
from torch.backends import cudnn
import utils as util
from CEM import CEM
import matplotlib.pyplot as plt
from torchsummary import summary

class Main:
    def __init__(self, mode="PN", max_iter=10, kappa=10, beta=1e-1, gamma=100,
                 data=MNIST, nn='MNIST_MNISTModel.pt', ae='MNIST_AE.pt',
                 c_steps=9, c_init=10., lr_init=1e-2, seed=None, report=True,
                 model_dir='models/', store_dir='results/'):
        """Initializing the main CEM attack module.

        Inputs:
        - mode      : search mode; "PN" or "PP"
        - max_iter  : maximum iterations running the attack
        - kappa     : confidence distance between label en max_nonlabel
        - beta      : fista regularizer
        - gamma     : weight of the ae
        - data      : dataloader to read images from
        - nn        : model ('black box' classifier)
        - ae        : autoencoder trained on data
        - c_steps   : amount of times to changes loss constant
        - c_init    : initial loss constant
        - lr_init   : initial learning rate of SGD
        - seed      : random seed
        - report    : print interations
        - model_dir : directory where models are stored
        - store_dir : directory to store images

        Returns: a MAIN object
        """
        dvc = 'cuda:0' if cuda.is_available() else 'cpu'
        self.set_seed(seed)

        #Load autoencoder and MNIST dataset.
        # AE_model = util.load_AE("mnist_AE_weights").to(dvc)
        self.ae = util.AE(torch.load(model_dir+ae, map_location=dvc)).to(dvc)
        self.nn = MNISTModel(torch.load(model_dir+nn, map_location=dvc)).to(dvc)
        self.data = data(dvc)
        self.mode = mode
        self.kappa = kappa
        self.gamma = gamma
        self.store = store_dir
        self.start, self.end = None, None

        self.cem = CEM(self.nn, mode, self.ae, lr_init=lr_init, c_init=c_init,
                       c_steps=c_steps, max_iterations=max_iter, kappa=kappa,
                       beta=beta, gamma=gamma, report=report)

    def set_seed(self, seed):
        """Set the random seeds."""
        if seed is not None:
            random.seed(seed)
            manual_seed(seed)
            if cuda.is_available():
                cudnn.deterministic = True
                cudnn.benchmark = False

    def set_image(self, id):
        """Load an image with id."""
        self.id = id
        image = self.data.test_data[id]
        self.pred, self.label, self.str = self.prediction(image)
        self.img, self.target = util.generate_data(self.data, id, self.label)
        print(f"Image:{id}, infer label:{self.label}")

    def prediction(self, data):
        """Do a prediction on data."""
        return util.model_prediction(self.nn, data)

    def attack(self):
        """Perform the attack."""
        # Create adversarial image from original image.
        self.adv = self.cem.attack(self.img, self.target).detach()
        delta = self.img-self.adv

        # Calculate probability classes for adversarial and delta image.
        self.adv_pred, self.adv_label, self.adv_str = self.prediction(self.adv)
        self.delta_pred, self.delta_label, self.delta_str = self.prediction(delta)
        self.delta = torch.abs(delta)-0.5

    def report(self):
        """Print report."""
        try:
            time = round(self.end - self.start, 1)
        except:
            time = 'None'

        INFO = f"\n\
                 [INFO]\n\
                 id:          {self.id}                           \n\
                 mode:        {self.mode}                         \n\
                 time (s):    {time}                              \n\
                 kappa:       {self.kappa}                        \n\
                 Original:    {self.label} {self.str}             \n\
                 Delta:       {self.delta_label} {self.delta_str} \n\
                 Adversarial: {self.adv_label} {self.adv_str}     \n"
        print(INFO)

    def store_images(self):
        """Store images to s.store directory."""
        suffix = f"id{self.id}_Orig{self.label}_Adv{self.adv_label}_Delta{self.delta_label}"
        save = f"{self.store}/{self.mode}_ID{self.id}_Gamma_{self.gamma}_Kappa_{self.kappa}"
        os.system(f"mkdir -p {save}")

        self.img_pic = util.save_img(self.img, f"{save}/Orig_{self.label}")
        self.delta_pic = util.save_img(self.delta, f"{save}/Delta_{suffix}", self.mode)
        self.adv_pic = util.save_img(self.img, f"{save}/Adv_{suffix}", self.mode, mode_img=self.delta)

    def show_images(self, w=18.5, h=10.5):
        """Show img, delta and adv next to each other."""
        f, ax = plt.subplots(1,3)
        f.set_size_inches(w, h)
        ax[0].imshow(self.img_pic, cmap='gray', vmin=0, vmax=255)
        ax[2].imshow(self.delta_pic)
        ax[1].imshow(self.adv_pic)
        plt.show()

    def summary(self):
        shape = (self.ae.image_size, self.ae.image_size, self.ae.num_channels)
        summary(self.ae, shape)
        summary(self.nn, shape)

        print(str(self.nn))
        print(str(self.ae))

    def run(self, id=2952, show=True):
        """Run the algorithm for specific image."""
        self.start = time.time()
        self.set_image(id)
        self.attack()
        self.end = time.time()
        self.report()
        self.store_images()
        if show:
            self.show_images()

pp = Main(mode='PP')
pp.summary()
# pn = Main(mode='PN')
# pp.run(1234)
# pn.run(1234)
