## (C) 2020 UvA FACT AI

import cem
import random

m = cem.Main(seed=121, type='FMNIST')

m.explain(3011, mode='PP', max_iter=2000, gamma=100, kappa=100)

# print(m.quant_eval(ids=random.sample(range(10000), 200)))
