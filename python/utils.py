#-*- coding: utf-8 -*-
# author: wangbb13
# create time: 2019-02-21 15:00
# reference: https://github.com/yusugomori/DeepLearning
import numpy as np
np.seterr(all='print') # set how floating-point errors are handled


def sigmoid(x):
	return 1. / (1 + np.exp(-x))


def dsigmoid(y):
	# y = sigmoid(x)
	# y' = y * (1 - y)
	return y * (1. - y)


def tanh(x):
	# hyperbolic tangent function (i.e. shuang qu zheng qie)
	# formula: (e^{2x} - 1) / (e^{2x} + 1)
	return np.tanh(x)


def dtanh(y):
	# derivative: y' = 1 - y^2
	return 1. - y * y


def softmax(x):
	# attention: how to prevent overflow
	e = np.exp(x - np.max(x))
	if e.ndim == 1:
		return e / np.sum(e, axis=0)
	else:
		return e / np.array([np.sum(e, axis=1)]).T  # dim = 2


def ReLU(x):
	return x * (x > 0)


def dReLU(x):
	return 1. * (x > 0)

