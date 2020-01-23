import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch

os.chdir("../Pytorch/")
sys.path.append("../Pytorch/")

import utils as util
from setup_mnist import MNIST, MNISTModel

#dvc = 'cuda:0' if cuda.is_available() else 'cpu'


data = MNIST()
sample1 = data.test_data[8]
print(sample1.shape)
sample = sample1.unsqueeze(0)
print(sample.shape)
print(data.test_labels[7])
AE_model_tf = util.load_AE("mnist_AE_weights")
AE_model = util.AE(restore=None)
AE_model.load_state_dict(torch.load("models/MNIST_AE.pt",  map_location=torch.device('cpu')))
decoded = AE_model(sample)
decoded_tf = AE_model_tf(sample)


fig1 = ((decoded + 0.5) * 255).round()
fig2 = ((decoded_tf + 0.5) * 255).round()
print(fig1.squeeze().detach().numpy().shape)

fig3 = np.around((decoded.detach().numpy()  + 0.5)*255)

fig3 = fig3.astype(np.uint8).squeeze()
print(fig3.shape)
print(fig1.detach().numpy()==fig3.T)
pic = Image.fromarray(fig3)
pic.save(name)
fig = plt.figure(figsize=(8, 8))
fig.add_subplot(1, 3, 1)
plt.imshow(fig1.squeeze().detach().numpy(), cmap='gray', vmin=0, vmax=255)
fig.add_subplot(1, 3, 2)
sample1 = ((sample1 + 0.5) * 255).round()
plt.imshow(sample1.reshape(28,28), cmap='gray', vmin=0, vmax=255)
fig.add_subplot(1, 3, 3)
plt.imshow(fig2.squeeze().detach().numpy().reshape(28,28), cmap='gray', vmin=0, vmax=255)
plt.show()
