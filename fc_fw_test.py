import numpy as np
from layers import FCLayer
from utils.tools import rel_error
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K
import warnings

warnings.filterwarnings('ignore')
inputs = np.random.uniform(size=(10, 20))
layer = FCLayer(in_features=inputs.shape[1], out_features=100)
out = layer.forward(inputs)

keras_model = models.Sequential()
keras_layer = layers.Dense(100, input_shape=inputs.shape[1:], use_bias=True, kernel_initializer='random_uniform', bias_initializer='zero')
# print (len(keras_layer.get_weights()))
keras_model.add(keras_layer)
sgd = optimizers.SGD(lr=0.01)
keras_model.compile(loss='mean_squared_error', optimizer='sgd')
keras_layer.set_weights([layer.weights, layer.bias])
keras_out = keras_model.predict(inputs, batch_size=inputs.shape[0])
print('Relative error (<1e-6 will be fine): ', rel_error(out, keras_out))