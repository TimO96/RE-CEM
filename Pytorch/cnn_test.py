from setup_mnist import MNIST, MNISTModel
from torch import load

mnist = MNIST()
cnn = MNISTModel(load("models/mnist.pt"))

pred = cnn.predict(mnist.test_data).argmax(dim=1
label = mnist.test_labels.argmax(dim=1)

acc = (pred == label).float().mean()

for name, param in cnn.named_parameters():
    print(name, param)

print(acc)
