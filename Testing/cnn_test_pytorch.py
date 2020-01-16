import os
import sys
os.chdir("../Pytorch/")
sys.path.append("../Pytorch/")

from setup_mnist import MNIST, MNISTModel
from torch import load, from_numpy
# from utils import h5_to_state_dict
from torchsummary import summary
import h5py
import numpy as np
import matplotlib.pylab as plt

weight_file = '../Original_Code/models/mnist'

def find(name, item):
    if isinstance(item, h5py.Dataset):
        print(name, item.shape)

f = h5py.File(weight_file, 'r')
f.visititems(find)

def h5_to_state_dict(h5_file, mapping):
    """Create Pytorch state_dict from h5 weight with mapping."""
    state_dict = {}
    with h5py.File(h5_file, 'r') as f:
        for h5, state in mapping.items():
            state_dict[state] = from_numpy(f[h5][:].T)
    return state_dict

# AE mapping
cnn_map1 = {
    'optimizer_weights/Variable_15:0' : 'model.0.bias',
    'optimizer_weights/Variable_14:0' : 'model.0.weight',
    'optimizer_weights/Variable_17:0' : 'model.2.bias',
    'optimizer_weights/Variable_16:0' : 'model.2.weight',
    'optimizer_weights/Variable_19:0' : 'model.5.bias',
    'optimizer_weights/Variable_18:0' : 'model.5.weight',
    'optimizer_weights/Variable_21:0' : 'model.7.bias',
    'optimizer_weights/Variable_20:0' : 'model.7.weight',
    'optimizer_weights/Variable_23:0' : 'model.11.bias',
    'optimizer_weights/Variable_22:0' : 'model.11.weight',
    'optimizer_weights/Variable_25:0' : 'model.13.bias',
    'optimizer_weights/Variable_24:0' : 'model.13.weight',
    'optimizer_weights/Variable_27:0' : 'model.15.bias',
    'optimizer_weights/Variable_26:0' : 'model.15.weight',
}

cnn_map = {
    'model_weights/conv2d_5/conv2d_5/bias:0'   : 'model.0.bias',
    'model_weights/conv2d_5/conv2d_5/kernel:0' : 'model.0.weight',
    'model_weights/conv2d_6/conv2d_6/bias:0'   : 'model.2.bias',
    'model_weights/conv2d_6/conv2d_6/kernel:0' : 'model.2.weight',
    'model_weights/conv2d_7/conv2d_7/bias:0'   : 'model.5.bias',
    'model_weights/conv2d_7/conv2d_7/kernel:0' : 'model.5.weight',
    'model_weights/conv2d_8/conv2d_8/bias:0'   : 'model.7.bias',
    'model_weights/conv2d_8/conv2d_8/kernel:0' : 'model.7.weight',
    'model_weights/dense_4/dense_4/bias:0'     : 'model.11.bias',
    'model_weights/dense_4/dense_4/kernel:0'   : 'model.11.weight',
    'model_weights/dense_5/dense_5/bias:0'     : 'model.13.bias',
    'model_weights/dense_5/dense_5/kernel:0'   : 'model.13.weight',
    'model_weights/dense_6/dense_6/bias:0'     : 'model.15.bias',
    'model_weights/dense_6/dense_6/kernel:0'   : 'model.15.weight',
}

cnn = MNISTModel(h5_to_state_dict(weight_file, cnn_map1))
summary(cnn, (cnn.image_size, cnn.image_size, cnn.num_channels), device="cpu")

mnist = MNIST()

plt.imshow(np.squeeze(mnist.test_data[4]))
#plt.show()
cnn.model.eval()
for name, param in cnn.model.named_parameters():
     param.requires_grad = False

for name, param in cnn.model.named_parameters():
    print(name, ':', param.requires_grad)

pred = cnn.predict(mnist.test_data[:100]).argmax(dim=1)
label = mnist.test_labels[:100].argmax(dim=1)

acc = (pred == label).float().mean()



# for name, param in cnn.named_parameters():
#     print(name, param)

print(acc)
