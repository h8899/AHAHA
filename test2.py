from layers import Convolution
import numpy as np
from utils.check_grads import check_grads_layer

batch = 10
conv_params={
'kernel_h': 3,
'kernel_w': 3,
'pad': 0,
'stride': 2,
'in_channel': 3,
'out_channel': 10
}
in_height = 10
in_width = 20
out_height = 1+(in_height+2*conv_params['pad']-conv_params['kernel_h'])//conv_params['stride']
out_width = 1+(in_width+2*conv_params['pad']-conv_params['kernel_w'])//conv_params['stride']
inputs = np.random.uniform(size=(batch, conv_params['in_channel'], in_height, in_width))
in_grads = np.random.uniform(size=(batch, conv_params['out_channel'], out_height, out_width))
conv = Convolution(conv_params)
check_grads_layer(conv, inputs, in_grads)