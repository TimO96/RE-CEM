#from cem.main import Main
# m = Main(mode='PP')
import cem
cem.Main(type='MNIST').explain(2953, mode='PP', gamma=0, kappa=10, max_iter=1000)
