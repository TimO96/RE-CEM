import os
import sys
os.chdir("../Original_Code/")
sys.path.append("../Original_Code/")

from setup_mnist import MNIST, MNISTModel
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt

mnist = MNIST()

plt.imshow(np.squeeze(mnist.test_data[4]))
plt.show()

with tf.Session() as sess:
    cnn = MNISTModel(restore="models/mnist", session=sess)
    
    pred = cnn.predict(mnist.test_data).eval().argmax(axis=1)
    label = mnist.test_labels.argmax(axis=1)
    # for var in tf.all_variables():
    #    print(var.eval())

    acc = (pred == label).mean()

print(acc)