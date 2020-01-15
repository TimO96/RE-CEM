import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import *
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

dl = torch.utils.data.DataLoader(
    datasets.MNIST('./data/mnist', train=True, download=True),
    batch_size=64, shuffle=True)

tensor = dl.dataset.data
tensor = tensor.to(dtype=torch.float32)
tr = tensor.reshape(tensor.size(0), -1)
tr = tr/128
targets = dl.dataset.targets
targets = targets.to(dtype=torch.long)

x_train = tr[0:50000]
y_train = targets[0:50000]
x_valid = tr[50000:60000]
y_valid = targets[50000:60000]

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=64, drop_last=False, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=64 * 2)

print(train_ds)
print(valid_ds)
