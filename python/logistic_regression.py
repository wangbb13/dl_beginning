# -*- coding: utf-8 -*-
# author: wangbb13
# create time: 2019-02-21 15:50
# reference: https://github.com/yusugomori/DeepLearning
import sys
import numpy as np
from utils import * 


class LogisticRegression(object):
	def __init__(self, _input, label, n_in, n_out):
		self.x = _input
		self.y = label
		self.W = np.zeros((n_in, n_out))
		self.b = np.zeros(n_out)
	
	def train(self, learning_rate=0.1, _input=None, L2_reg=0.0, reduce_rate=0.995, nepochs=500):
		if _input is not None:
			self.x = _input

		lr = learning_rate
		for _ in range(nepochs):
			p_y_given_x = self.predict(self.x)
			print('epoch', _, 'cost:', self.negative_log_likelihood(p_y_given_x))
			d_y = self.y - p_y_given_x
			self.W += lr * np.dot(self.x.T, d_y) - lr * L2_reg * self.W
			self.b += lr * np.mean(d_y, axis=0)
			lr *= reduce_rate
	
	def predict(self, x):
		return softmax(np.dot(x, self.W) + self.b)
	
	def negative_log_likelihood(self, pred_y):
		return -np.mean(np.sum(self.y * np.log(pred_y) + (1 - self.y) * np.log(1 - pred_y), axis=1))


def test(learning_rate=0.1, nepochs=500):
	rng = np.random.RandomState(123)
	# construct training data
	d = 2
	N = 10
	x1 = rng.randn(N, d) + np.array([0,0])
	x2 = rng.randn(N, d) + np.array([20, 10])
	y1 = [[1, 0] for _ in range(N)]
	y2 = [[0, 1] for _ in range(N)]
	x = np.r_[x1.astype(int), x2.astype(int)]
	y = np.r_[y1, y2]

	# build model
	classifier = LogisticRegression(x, y, d, 2)
	classifier.train(learning_rate=learning_rate, nepochs=nepochs)

	# results
	result = classifier.predict(x)
	for i in range(N):
		print(result[i])
	print('')
	for i in range(N):
		print(result[N+i])


if __name__ == '__main__':
	test()

