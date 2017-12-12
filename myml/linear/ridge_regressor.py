import numpy as np
from .regressor import Regressor

class RidgeRegressor(Regressor):

	def __init__(self,alpha = 0.5):
		self.alpha = alpha

	def _fit(self, X, t):
		eye = np.eye(np.size(X, 1))
		self.w = np.linalg.solve(self.alpha * eye + np.matmul(X.T,X), X.T @ t)

	def _predict(self, X):
		y = np.matmul(X,self.w)
		return y
