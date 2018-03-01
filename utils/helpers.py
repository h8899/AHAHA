import numpy as np
import math
from scipy import ndimage

def augment(mnist):
	new_shape = np.array(list(mnist.x_train.shape))
	new_shape[0] = 2 * new_shape[0]
	augmented_train_datas = np.zeros(new_shape)
	augmented_train_labels = np.zeros([2 * mnist.x_train.shape[0]], dtype=int) 
	for i in range(mnist.x_train.shape[0]):
		img = np.array(list(mnist.x_train[i, 0, :, :]))
		background_value = np.median(img)

		# Shift
		shifted_img = ndimage.shift(img, np.random.randint(-2, 2, 2), cval=background_value)

		# Rotate
		rotated_img = ndimage.rotate(img, np.random.randint(-30, 30), reshape=False, cval=background_value)

		augmented_train_datas[2*i, 0, :, :] = shifted_img

		augmented_train_datas[2*i + 1, 0, :, :] = rotated_img

		augmented_train_labels[2*i] = int(mnist.y_train[i])

		augmented_train_labels[2*i+1] = int(mnist.y_train[i])

	mnist.x_train = np.append(mnist.x_train, augmented_train_datas, axis=0)

	mnist.y_train = np.append(mnist.y_train, augmented_train_labels, axis=0)

	
	new_shape = np.array(list(mnist.x_test.shape))
	new_shape[0] = 2 * new_shape[0]
	augmented_test_datas = np.zeros(new_shape)
	augmented_test_labels = np.zeros([2 * mnist.x_test.shape[0]], dtype=int) 
	for i in range(mnist.x_test.shape[0]):
		img = np.array(list(mnist.x_test[i, 0, :, :]))
		background_value = np.median(img)

		# Shift
		shifted_img = ndimage.shift(img, np.array([-1, 1]), cval=background_value)

		# Rotate
		rotated_img = ndimage.rotate(img, 15, reshape=False, cval=background_value)

		augmented_test_datas[2*i, 0, :, :] = shifted_img

		augmented_test_datas[2*i + 1, 0, :, :] = rotated_img

		augmented_test_labels[2*i] = int(mnist.y_test[i])

		augmented_test_labels[2*i+1] = int(mnist.y_test[i])

	mnist.x_test = np.append(mnist.x_test, augmented_test_datas, axis=0)

	mnist.y_test = np.append(mnist.y_test, augmented_test_labels, axis=0)	



