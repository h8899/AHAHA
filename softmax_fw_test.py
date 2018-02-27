import numpy as np
from loss import SoftmaxCrossEntropy
from utils.tools import rel_error
import keras
from keras import layers
from keras import models 
from keras import optimizers
from keras import backend as K
import warnings

warnings.filterwarnings('ignore')
batch = 10
num_class = 10
inputs = np.random.uniform(size=(batch, num_class))
targets = np.random.randint(num_class, size=batch)
loss = SoftmaxCrossEntropy(num_class)
out, _ = loss.forward(inputs, targets)


keras_inputs = K.softmax(inputs)
keras_targets = np.zeros(inputs.shape, dtype='int')

for i in range(batch):
	keras_targets[i, targets[i]] = 1

keras_out = K.mean(K.categorical_crossentropy(keras_targets, keras_inputs, from_logits=False))
print('Relative error (<1e-6 will be fine): ', rel_error(out, K.eval(keras_out)))