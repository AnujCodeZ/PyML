import numpy as np


class SVM:

    def __init__(self, features, lambda_param):
        self.lambda_param = lambda_param
        self.weights = np.zeros(features)
        self.bias = 0
        self.dw = np.zeros_like(self.weights)
        self.db = 0

    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        y_hat = np.where(z >= 0, 1., 0.)
        return z, y_hat

    def cost(self, y, z):
        cost_1 = np.where(z>=1, 0., z)
        cost_1 = np.where(z<=-1, 0., z)
        loss = (np.dot(y.T, cost_1)
                np.dot((1 - y).T, cost_0))
        loss = loss / self.lambda_param
        loss = loss + (1/2) * np.dot(self.weights.T, self.weights)
        return loss
    
    def backward(self, x, y, y_hat, z):
        if z >= 1 or z <= -1:
            self.dw = self.weights
            self.db = 0
        else:
            self.dw = np.dot(x.T, ((-y/z) + (1 - y)/(1-z))) + \
                self.weights
            self.db = np.sum(((-y/z) + (1 - y)/(1-z)), axis=1)
    
        
        
        