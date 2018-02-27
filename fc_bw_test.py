from layers import FCLayer
import numpy as np
from utils.check_grads import check_grads_layer

batch = 10

inputs = np.random.uniform(size=(10, 20))
out_features = 100
layer = FCLayer(in_features=inputs.shape[1], out_features=out_features)

in_grads = np.random.uniform(size=(batch, out_features))
check_grads_layer(layer, inputs, in_grads)