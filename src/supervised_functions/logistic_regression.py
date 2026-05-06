"""
Logistic Regression — binary classification via Gradient Descent.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '_shared'))

from _linear_helpers import sigmoid, binary_cross_entropy
from gradient_descent import GradientDescentMixin


class LogisticRegression(GradientDescentMixin):
    def __init__(self, learning_rate=0.01, n_iterations=1000,
                 batch_size=None, threshold=0.5, l2=0.0, verbose=False):
        self.learning_rate=learning_rate; self.n_iterations=n_iterations
        self.batch_size=batch_size; self.threshold=threshold; self.l2=l2; self.verbose=verbose

    def fit(self, X, y):
        X,y = np.asarray(X,dtype=float), np.asarray(y,dtype=float)
        self._gd_fit(X,y,self.learning_rate,self.n_iterations,self.batch_size,self.verbose)
        return self

    def _compute_loss(self,X,y,w,b):
        loss = binary_cross_entropy(y, sigmoid(X@w+b))
        if self.l2>0: loss += (self.l2/(2*len(y)))*np.sum(w**2)
        return loss

    def _compute_gradients(self,X,y,w,b):
        n=len(y); e=sigmoid(X@w+b)-y
        return X.T@e/n + (self.l2/n)*w, np.sum(e)/n

    def predict_proba(self,X): return sigmoid(np.asarray(X,dtype=float)@self.weights_+self.bias_)
    def predict(self,X): return (self.predict_proba(X)>=self.threshold).astype(int)
    def accuracy(self,X,y): return float(np.mean(self.predict(X)==np.asarray(y,dtype=float)))
    def __repr__(self): return f"LogisticRegression(lr={self.learning_rate}, l2={self.l2})"
