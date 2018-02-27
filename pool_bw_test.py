from layers import Pooling
import numpy as np
from utils.check_grads import check_grads_layer

batch = 10
params={
	'pool_type': 'max',
	'pool_height': 5,
	'pool_width': 5,
	'pad': 0,
	'stride': 2,
}
in_channel = 5

in_height = 10
in_width = 20
out_height = 1+(in_height+2*params['pad']-params['pool_height'])//params['stride']
out_width = 1+(in_width+2*params['pad']-params['pool_width'])//params['stride']
inputs = np.random.uniform(size=(batch, in_channel, in_height, in_width))
in_grads = np.random.uniform(size=(batch, in_channel, out_height, out_width))
pool = Pooling(params)
check_grads_layer(pool, inputs, in_grads)