## (C) 2020 UvA FACT AI

import cem
import random

m = cem.Main(seed=121, type='MNIST')

print(m.quant_eval(ids=random.sample(range(10000), 200)))


