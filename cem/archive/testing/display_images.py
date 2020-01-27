import os
import sys
os.chdir("../Pytorch/")
sys.path.append("../Pytorch/")

from setup_mnist import *
from utils import AE, save_img, load_AE

ae = AE()
ae.load_state_dict(load("models/MNIST_AE.pt"))

ae_2 = load_AE("mnist_AE_weights")

mnist = MNIST(device="cpu")
test_imgs = mnist.test_data[:5]
reconstruced_imgs = ae(test_imgs)
reconstruced_imgs_2 = ae_2(test_imgs)

os.chdir("../Testing/")
os.makedirs("./images", exist_ok=True)
for i in range(len(test_imgs)):
    save_img(test_imgs[i], "images/Original_"+str(i+1), save_tensor=False)
    save_img(reconstruced_imgs[i], "images/Reconstructed_"+str(i+1), save_tensor=False)
    save_img(reconstruced_imgs_2[i], "images/Reconstruced_Loaded"+str(i+1), save_tensor=False)

