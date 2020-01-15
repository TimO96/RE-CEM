from setup_mnist import MNIST, MNISTModel
import tensorflow as tf

mnist = MNIST()

with tf.Session() as sess:
    cnn = MNISTModel(restore="models/mnist", session=sess)
    pred = cnn.predict(mnist.test_data).eval().argmax(axis=1)
    label = mnist.test_labels.argmax(axis=1)


acc = (pred == label).mean()

print(acc)