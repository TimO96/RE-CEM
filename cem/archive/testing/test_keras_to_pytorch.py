import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt

#os.chdir("../Original_Code")
#sys.path.append("../Original_Code/")

from test_setup_mnist_tf import MNIST, MNISTModel



mnist = MNIST()

with tf.Session() as sess:
    cnn = MNISTModel(restore=None, session=sess)
    cnn.model.load_weights("../Original_Code/models/mnist")
    weights = cnn.model.get_weights()
    cnn.model2.layers[0].set_weights([weights[0], weights[1]])
    cnn.model2.layers[2].set_weights([weights[2], weights[3]])
    cnn.model2.layers[5].set_weights([weights[4], weights[5]])
    cnn.model2.layers[7].set_weights([weights[6], weights[7]])
    cnn.model2.layers[11].set_weights([weights[8], weights[9]])

    #cnn.model3.layers[0].set_weights([weights[8], weights[9]])
    dict_weights = {}
    for layer in cnn.model.layers:
        dict_weights[layer.get_config()['name']] = layer.get_weights()

    #pred = cnn.predict(mnist.test_data[:100]).eval().argmax(axis=1)
    pred1 = cnn.predict(mnist.test_data[:1]).eval()
    label = mnist.test_labels[:100].argmax(axis=1)
    # for var in tf.all_variables():
    #    print(var.eval())

    #acc = (pred == label).mean()

#print(acc)


from torch import load, from_numpy, tensor, float32, flip, rot90
import torch
# from utils import h5_to_state_dict
from torchsummary import summary


from test_setup_mnist import MNIST, MNISTModel

cnn2 = MNISTModel()



cnn2.cv1.weight.data = tensor(weights[0].transpose(3, 2, 0, 1), dtype=float32)
cnn2.cv1.bias.data= tensor(weights[1], dtype=float32)
cnn2.cv2.weight.data= tensor(weights[2].transpose(3, 2, 0, 1), dtype=float32)
cnn2.cv2.bias.data=tensor(weights[3], dtype=float32)
cnn2.cv3.weight.data=tensor(weights[4].transpose(3, 2, 0, 1), dtype=float32)
cnn2.cv3.bias.data=tensor(weights[5], dtype=float32)
cnn2.cv4.weight.data=tensor(weights[6].transpose(3, 2, 0, 1), dtype=float32)
cnn2.cv4.bias.data=tensor(weights[7], dtype=float32)
cnn2.fc1.weight.data=tensor(weights[8].T, dtype=float32)
cnn2.fc1.bias.data=tensor(weights[9], dtype=float32)
cnn2.fc2.weight.data=tensor(weights[10].T, dtype=float32)
cnn2.fc2.bias.data=tensor(weights[11], dtype=float32)
cnn2.fc3.weight.data=tensor(weights[12].T, dtype=float32)
cnn2.fc3.bias.data=tensor(weights[13], dtype=float32)


#pred2 = cnn2.predict(from_numpy(mnist.test_data[:100])).argmax(dim=1)
pred2 = cnn2.predict(tensor(mnist.test_data[:1]).float())

from test_setup_mnist_tf import MNIST, MNISTModel


print(pred1.shape)

print(pred2.shape)
print(pred1)
print(pred2)

torch.set_printoptions(threshold=10000)

diff = pred1 - pred2
print(diff)
print(diff.shape)
print(diff.sum())

#acc = (pred2 == from_numpy(label)).float().mean()
