## main.py -- Main class in which the code can be run from its functions.

## (C) 2020 Changes by UvA FACT AI group [Pytorch conversion]

## Based on:
## Copyright (C) 2018, IBM Corp
##                     Chun-Chen Tu <timtu@umich.edu>
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>

import os
import random
import time
import matplotlib.pyplot as plt
from torchsummary import summary
from torch import cuda, manual_seed, load, abs
from torch import abs as torchabs
from torch.backends import cudnn

from setup_mnist import MNIST, MNISTModel
import utils as util
from CEM import CEM


class Main:
    def __init__(self, mode="PN", max_iter=10, kappa=10, beta=1e-1, gamma=100,
                 data=MNIST, nn='MNISTModel.pt', ae='AE.pt', type='MNIST',
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
            - type      : dataset type ("MNIST" or "FMNIST")
            - c_steps   : amount of times to changes loss constant
            - c_init    : initial loss constant
            - lr_init   : initial learning rate of SGD
            - seed      : random seed
            - report    : print interations
            - model_dir : directory where models are stored
            - store_dir : directory to store images
            - darg      : optional arguments for data
        Returns:
            - a MAIN object instance
        """
        # Initialize CPU/GPU device and set seed for reproducibility.
        dvc = 'cuda:0' if cuda.is_available() else 'cpu'
        self.dvc = dvc
        self.set_seed(seed)

        # Load autoencoder and the CNN for the MNIST dataset.
        type_dir = model_dir + type + '_'
        self.ae = util.AE(load(type_dir+ae, map_location=dvc)).to(dvc)
        self.nn = MNISTModel(load(type_dir+nn, map_location=dvc)).to(dvc)

        # Initialize dataset and class variables.
        self.data = data(dvc, type)
        self.mode = mode
        self.kappa = kappa
        self.gamma = gamma
        self.store = store_dir
        self.start, self.end = None, None

        # Intialize CEM class for attack and perform PP and PN analysis.
        self.cem = CEM(self.nn, mode, self.ae, lr_init=lr_init, c_init=c_init,
                       c_steps=c_steps, max_iterations=max_iter, kappa=kappa,
                       beta=beta, gamma=gamma, report=report)

    def set_seed(self, seed):
        """Set the random seeds (numpy and pytorch) for reproducibility."""
        if seed is not None:
            random.seed(seed)
            manual_seed(seed)
            if self.dvc == 'cuda:0':
                cudnn.deterministic = True
                cudnn.benchmark = False

    def set_image(self, id):
        """
        Load an image with id and retrieve ground truth label from
        the black box model f self.nn.
        """
        self.id = id
        image = self.data.test_data[id]
        self.pred, self.label, self.str = self.prediction(image)
        self.img, self.target = util.generate_data(self.data, id, self.label)
        print(f"Image:{id}, infer label:{self.label}")

    def prediction(self, data):
        """Perform a prediction on the data based on the neural network."""
        return util.model_prediction(self.nn, data)

    def attack(self):
        """Perform the attack."""
        # Create adversarial image from original image.
        self.adv = self.cem.attack(self.img, self.target).detach()
        delta = self.img - self.adv

        # Calculate probability classes for adversarial and delta image.
        self.adv_pred, self.adv_label, self.adv_str = self.prediction(self.adv)
        self.delta_pred, self.delta_label, self.delta_str = \
                                                          self.prediction(delta)
        # Perform appropriate scaling.
        self.delta = abs(delta) - 0.5

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
        # sys.stdout.flush()

    def store_images(s):
        """Store images to s.store directory."""
        sfx = f"id{s.id}_Orig{s.label}_Adv{s.adv_label}_Delta{s.delta_label}"
        dir = f"{s.store}/{s.mode}_ID{s.id}_Gamma_{s.gamma}_Kappa_{s.kappa}"
        os.system(f"mkdir -p {dir}")

        s.img_pic = util.save_img(s.img, f"{dir}/Orig_{s.label}")
        s.delta_pic = util.save_img(s.delta, f"{dir}/Delta_{sfx}", s.mode)
        s.adv_pic = util.save_img(s.img, f"{dir}/Adv_{sfx}", s.mode, s.delta)

    def show_images(self, w=18.5, h=10.5):
        """Show img, delta and adv next to each other."""
        f, ax = plt.subplots(1,3)
        f.set_size_inches(w, h)
        ax[0].imshow(self.img_pic, cmap='gray', vmin=0, vmax=255)
        ax[1].imshow(self.delta_pic)
        ax[2].imshow(self.adv_pic)
        plt.show()

    def summary(self):
        """Print information on structure of the model and the autoencoder."""
        shape = (self.ae.image_size, self.ae.image_size, self.ae.num_channels)
        print(str(self.ae))
        summary(self.ae, shape)
        print(str(self.nn))
        summary(self.nn, shape)

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

if __name__ == "__main__":
    pp = Main(mode='PP')
    pn = Main(mode='PN')
    pp.summary()
    pp.run(1234)
    pn.run(1234)
