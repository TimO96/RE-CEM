# main.py -- Main class in which the code can be run from its functions.

# (C) 2020 Changes by UvA FACT AI group [Pytorch conversion]

# Based on:
# Copyright (C) 2018, IBM Corp
#                     Chun-Chen Tu <timtu@umich.edu>
#                     PaiShun Ting <paishun@umich.edu>
#                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>

import os
import random
import time
import matplotlib.pyplot as plt

from torchsummary import summary
from torch import cuda, manual_seed, load, abs
from torch.backends import cudnn

from .models.models import MNISTModel, AE
from .data.data import MNIST
from .attack import Attack
from . import models
from . import utils as util


class Main:
    def __init__(self, type='MNIST', nn='MNISTModel.pt', ae='AE.pt',
                 mode=None, seed=None):
        """Initialize the CEM controller.
            - type : dataset type; MNIST or FMNIST
            - mode : search mode; "PN" or "PP"
            - seed : random seed
            - nn        : model ('black box' classifier)
            - ae        : autoencoder trained on data
            - model_dir : directory where models are stored
        """
        dvc = 'cuda:0' if cuda.is_available() else 'cpu'
        type = type.upper()
        assert type in ('MNIST', 'FMNIST'), f"Unsupported type {type}."

        self.dvc = dvc
        self.data = MNIST(dvc, type)
        self.set_seed(seed)

        # Load autoencoder and the CNN for the MNIST dataset.
        type_dir = os.path.dirname(models.__file__) + '/' + type + '_'
        self.ae = AE(load(type_dir+ae, map_location=dvc)).to(dvc)
        self.nn = MNISTModel(load(type_dir+nn, map_location=dvc)).to(dvc)

        if mode is not None:
            self.set_mode(mode)

    def set_seed(self, seed):
        """Set the random seeds (python and pytorch) for reproducibility."""
        if seed is not None:
            random.seed(seed)
            manual_seed(seed)
            if self.dvc == 'cuda:0':
                cudnn.deterministic = True
                cudnn.benchmark = False

    def set_mode(self, mode):
        """Set internal mode."""
        assert mode in ("PP", "PN"), "Expected mode 'PP' or 'PN'."
        self.mode = mode

    def model_summary(self):
        """Print information on structure of the model and the autoencoder."""
        shape = (self.ae.image_size, self.ae.image_size, self.ae.num_channels)
        print(str(self.ae))
        summary(self.ae, shape)
        print(str(self.nn))
        summary(self.nn, shape)

    def explain(self, image_id, mode=None, show=True, **args):
        """Explain a specific id in mode.

        Input:
        - id     : dataset image id
        - mode   : search mode; "PN" or "PP"
        - object : return CEM object
        - show   : show explained images
        """
        if mode is None:
            mode = self.mode

        explain = CEM(nn=self.nn, ae=self.ae, dvc=self.dvc, mode=mode, **args)
        explain.run(self.data, image_id, show)

        return explain

    def quant_eval(self, ids, **kwargs):
        """Quantative evaluation"""
        assert iter(ids), "Ids should be an iterable."

        explain = CEM(nn=self.nn, ae=self.ae, dvc=self.dvc, mode=self.mode,
                      report=False, **kwargs)

        score = []
        for mode in ["PP", "PN"]:
            for image_id in ids:
                score.append(explain.match_labels(self.data, image_id, mode))

        return sum(score)/len(score)

    def show_array(self, id, w=18.5, h=10.5, **kwargs):
        """Show arrayplot for id with paper combinations of arguments."""
        f, ax = plt.subplots(1, 5)
        f.set_size_inches(w, h)
        n = 1

        # Add adversarial PP/PN with gamma 0/100 to subplot.
        for mode in ['PP', 'PN']:
            for gamma in [0, 100]:
                cem = self.explain(id, mode=mode, gamma=gamma, show=False,
                                   report=False, **kwargs)
                ax[n].imshow(cem.adv_pic)
                ax[n].set_title(f"{mode}, $\gamma$={gamma} ({cem.adv_label})")
                n += 1

        # Add original image.
        ax[0].imshow(cem.img_pic, cmap='gray', vmin=0, vmax=255)
        ax[0].set_title(f"Original image ({cem.label})")
        [axis.set_axis_off() for axis in ax.ravel()]
        plt.show()


class CEM:
    def __init__(self, nn, ae, dvc, mode="PN", max_iter=1000, kappa=10,
                 beta=1e-1, gamma=100, c_steps=9, c_init=10., lr_init=1e-2,
                 report=True, store_dir='results/'):
        """Initializing the main CEM attack module.
        Inputs:
            - max_iter  : maximum iterations running the attack
            - kappa     : confidence distance between label en max_nonlabel
            - beta      : fista regularizer
            - gamma     : weight of the ae
            - nn        : model ('black box' classifier)
            - ae        : autoencoder trained on data
            - c_steps   : amount of times to changes loss constant
            - c_init    : initial loss constant
            - lr_init   : initial learning rate of SGD
            - report    : print interations
            - store_dir : directory to store images
            - dvc       : tensor device (cuda or cpu)
        Returns:
            - a CEM object instance
        """
        # Initialize dataset and class variables.
        self.dvc = dvc
        self.ae = ae
        self.nn = nn
        self.mode = mode
        self.kappa = kappa
        self.gamma = gamma
        self.store = store_dir
        self.start, self.end = None, None

        # Intialize CEM class for attack and perform PP and PN analysis.
        self.cem_att = Attack(self.nn, self.ae, lr_init=lr_init, c_init=c_init,
                              c_steps=c_steps, max_iterations=max_iter,
                              kappa=kappa, beta=beta, gamma=gamma,
                              report=report)

    def set_image(self, data, image_id):
        """
        Load an image with id and retrieve ground truth label from
        the black box model f self.nn.
        """
        self.id = image_id
        image = data.test_data[image_id]
        self.pred, self.label, self.str = self.prediction(image)
        self.img, self.target = util.generate_data(data, image_id, self.label)
        print(f"Image:{image_id}, infer label:{self.label}")

    def prediction(self, data):
        """Perform a prediction on the data based on the neural network."""
        return util.model_prediction(self.nn, data)

    def attack(s, mode=None):
        """Perform the attack."""
        if not mode:
            mode = s.mode
        s.start = time.time()
        # Create adversarial image from original image.
        s.adv = s.cem_att.attack(s.img, s.target, mode).detach()
        delta = s.img - s.adv

        # Calculate probability classes for adversarial and delta image.
        s.adv_pred, s.adv_label, s.adv_str = s.prediction(s.adv)
        s.delta_pred, s.delta_label, s.delta_str = s.prediction(delta)

        # Perform appropriate scaling.
        s.delta = abs(delta) - 0.5
        s.end = time.time()

    def match_labels(self, data, image_id, mode):
        """Return whether labels match or not"""
        self.set_image(data, image_id)
        self.attack(mode=mode)

        if mode == "PN":
            return self.adv_label != self.label

        return self.delta_label == self.label

    def report(self):
        """Print report."""
        try:
            time = round(self.end - self.start, 1)
        except Exception:
            time = 'None'

        info = f"\n\
        [INFO]\n\
        id:          {self.id}                           \n\
        mode:        {self.mode}                         \n\
        time (s):    {time}                              \n\
        kappa:       {self.kappa}                        \n\
        gamma:       {self.gamma}                        \n\
        Original:    {self.label} {self.str}             \n\
        Delta:       {self.delta_label} {self.delta_str} \n\
        Adversarial: {self.adv_label} {self.adv_str}     \n"
        print(info)

    def store_images(s):
        """Store images to s.store directory."""
        sfx = f"id{s.id}_Orig{s.label}_Adv{s.adv_label}_Delta{s.delta_label}"
        s_dir = f"{s.store}/{s.mode}_ID{s.id}_Gamma_{s.gamma}_Kappa_{s.kappa}"
        os.system(f"mkdir -p {dir}")

        s.img_pic = util.save_img(s.img, f"{s_dir}/Orig_{s.label}")
        s.delta_pic = util.save_img(s.delta, f"{s_dir}/Delta_{sfx}", s.mode)
        s.adv_pic = util.save_img(s.img, f"{s_dir}/Adv_{sfx}", s.mode, s.delta)

    def show_images(self, w=18.5, h=10.5):
        """Show img, delta and adv next to each other."""
        f, ax = plt.subplots(1, 3)
        f.set_size_inches(w, h)
        ax[0].imshow(self.img_pic, cmap='gray', vmin=0, vmax=255)
        ax[1].imshow(self.delta_pic)
        ax[2].imshow(self.adv_pic)
        [axis.set_axis_off() for axis in ax.ravel()]
        plt.show()

    def run(self, data, image_id=2952, show=True):
        """Run the algorithm for specific image."""
        self.set_image(data, image_id)
        self.attack()
        self.report()
        self.store_images()
        if show:
            self.show_images()
