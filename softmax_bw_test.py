from loss import SoftmaxCrossEntropy
import numpy as np
from utils.check_grads import check_grads_loss

batch = 10
num_class = 10

inputs = np.random.uniform(size=(batch, num_class))
targets = np.random.randint(num_class, size=batch)
loss = SoftmaxCrossEntropy(num_class)

check_grads_loss(loss, inputs, targets)