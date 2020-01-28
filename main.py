## (C) 2020 UvA FACT AI

import cem
m = cem.Main(seed=121, type='MNIST')

print(m.quant_eval(ids=list(range(100))))


