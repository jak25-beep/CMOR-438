"""
Linear Regression — OLS, Ridge, and Gradient Descent.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))

from _linear_helpers import ols_normal_equation, ridge_normal_equation, mean_squared_error, r_squared
from gradient_descent import GradientDescentMixin


class LinearRegression(GradientDescentMixin):
    """
    Parameters
    ----------
    method : 'ols' | 'ridge' | 'gd'
    alpha : L2 regularisation strength (Ridge only)
    learning_rate : step size (GD only)
    n_iterations : number of epochs (GD only)
    batch_size : None=full batch, 1=SGD, k=mini-batch (GD only)
    verbose : bool
    """
    def __init__(self, method="ols", alpha=1.0, learning_rate=0.01,
                 n_iterations=1000, batch_size=None, verbose=False):
        assert method in ("ols","ridge","gd")
        self.method=method; self.alpha=alpha; self.learning_rate=learning_rate
        self.n_iterations=n_iterations; self.batch_size=batch_size; self.verbose=verbose
        self.weights_=None; self.bias_=0.0

    def fit(self, X, y):
        X,y = np.asarray(X,dtype=float), np.asarray(y,dtype=float)
        if self.method=="ols":   self.weights_,self.bias_ = ols_normal_equation(X,y)
        elif self.method=="ridge": self.weights_,self.bias_ = ridge_normal_equation(X,y,self.alpha)
        elif self.method=="gd":  self._gd_fit(X,y,self.learning_rate,self.n_iterations,self.batch_size,self.verbose)
        return self

    def _compute_loss(self,X,y,w,b): return mean_squared_error(y, X@w+b)
    def _compute_gradients(self,X,y,w,b):
        n=len(y); e=X@w+b-y
        return (2/n)*X.T@e, (2/n)*np.sum(e)

    def predict(self,X): return np.asarray(X,dtype=float)@self.weights_+self.bias_
    def score(self,X,y): return r_squared(np.asarray(y,dtype=float), self.predict(X))
    def mse(self,X,y):   return mean_squared_error(np.asarray(y,dtype=float), self.predict(X))
    def __repr__(self): return f"LinearRegression(method='{self.method}')"
