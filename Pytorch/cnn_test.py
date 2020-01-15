from setup_mnist import MNIST, MNISTModel

mnist = MNIST()
cnn = MNISTModel(restore="models/mnist.pt")

pred = cnn.predict(mnist.test_data).argmax(dim=1)
label = mnist.test_labels.argmax(dim=1)

acc = (pred == label).float().mean()

print(acc)