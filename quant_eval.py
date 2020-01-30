## (C) 2020 UvA FACT AI

import cem
import random

m = cem.Main(seed=121, type='MNIST')

# m.show_array(12, max_iter=1000)

print(m.quant_eval(ids=random.sample(range(10000), 200)))
