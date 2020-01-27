#from cem.main import Main
# m = Main(mode='PP')
import cem
# cem.Main(mode='PP', type='FMNIST').explain(10, gamma=0, kappa=10, max_iter=1000)

cem.Main(type='MNIST').show_array(10, kappa=10, max_iter=10)
