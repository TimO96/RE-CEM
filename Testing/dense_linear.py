from keras.models import Sequential
from keras.layers import Dense, Flatten

import torch
import torch.nn as nn
import numpy as np

IN = 1024
OUT = 200
B = 100

tfnn = Sequential()
tfnn.add(Flatten())
tfnn.add(Dense(OUT, input_dim=IN, activation='linear'))
tfnn.load_weights("../Original_Code/models/mnist", by_name=True)
weights = tfnn.get_weights()
print(weights)

ptnn = nn.Linear(IN,OUT)

#weights = np.ones((IN,OUT))
#bias = np.zeros((OUT,))
x = np.arange(B*IN).reshape(B,IN)

ptnn.weight.data = torch.tensor(weights[0].T, dtype=torch.float)
ptnn.bias.data = torch.tensor(weights[1], dtype=torch.float)

pred_tf = tfnn.predict(x)
pred_pt = ptnn(torch.tensor(x.view(-1,1024), dtype=torch.float))

print(pred_tf.shape)
print(pred_pt.shape)
print(pred_tf)
print(pred_pt)

diff = torch.tensor(pred_tf) - pred_pt
print(diff)
