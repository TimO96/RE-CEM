import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np

weights = ((np.array(range(9))+1)/10).reshape(3,3)

class ConvTestPT(nn.Module):
    def __init__(self):
        super(ConvTestPT, self).__init__()

        self.model = nn.Conv2d(1, 1, (3,3), bias=False)
        self.model.weight.data = torch.from_numpy(weights).float().view(1,1,3,3)
    
    def forward(self, input):
        return self.model(input)

class ConvTestTF:
    def __init__(self):
        self.model = tf.keras.layers.Conv2D(1, (3,3), weights=weights.reshape(1,3,3,1,1), input_shape=(28,28,1), use_bias=False)

    def forward(self, input):
        return self.model(input)

pt_model = ConvTestPT()
tf_model = ConvTestTF()

test_input = np.ones((9,9))
with tf.Session() as sess:
    print(pt_model(torch.from_numpy(test_input).float().view(1,1,9,9)).detach().numpy())

    print(tf_model.forward(test_input.reshape(1,9,9,1)).eval())