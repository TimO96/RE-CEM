from setup_mnist import MNIST, MNISTModel
import numpy as np
import matplotlib.pylab as plt

mnist = MNIST()

plt.imshow(np.squeeze(mnist.test_data[4]))
plt.show()

cnn = MNISTModel(restore="models/mnist.pt")

pred = cnn.predict(mnist.test_data).argmax(dim=1)
label = mnist.test_labels.argmax(dim=1)
print(label)

acc = (pred == label).float().mean()

# for name, param in cnn.named_parameters():
#     print(name, param)

print(acc)
