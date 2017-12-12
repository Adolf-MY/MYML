import numpy as np
from .regressor import Regressor

class LinearRegressor(Regressor):

	def _fit(self,X,t):
		self.w  = np.matmul(np.linalg.pinv(X),t)
		self.var = np.mean(np.square(np.matmul(X,self.w)-t))

	def _predict(self, X, return_std=False):
		y = np.matmul(X,self.w)
		if return_std:
			y_std = np.sqrt(self.var) + np.zeros_like(y)
			return y, y_std
		return y
