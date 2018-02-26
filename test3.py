from layers import Dropout
import numpy as np
from utils.check_grads import check_grads_layer

batch = 10
ratio = 0.1
height = 10
width = 20
channel = 10
np.random.seed(1234)
inputs = np.random.uniform(size=(batch, channel, height, width))
in_grads = np.random.uniform(size=(batch, channel, height, width))
dropout = Dropout(ratio, seed=1234)
dropout.set_mode(True)
check_grads_layer(dropout, inputs, in_grads)