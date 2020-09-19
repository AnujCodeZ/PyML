import numpy as np
from tqdm import tqdm


class SVM:

    def __init__(self, features, lambda_param=0.01):
        self.lambda_param = lambda_param
        self.weights = np.zeros(features)
        self.bias = 0
        self.dw = np.zeros_like(self.weights)
        self.db = 0
    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return np.sign(z)
    
    def backward(self, x_i, y_i):
        condition = y_i * (np.dot(x_i, self.weights)-self.bias) >= 1
        if condition:
            self.dw = 2 * self.lambda_param * self.weights
            self.db = 0
        else:
            self.dw = 2 * self.lambda_param * self.weights - \
                np.dot(x_i, y_i)
            self.db = y_i
    
    def update(self, lr):
        self.weights = self.weights - lr * self.dw
        self.bias = self.bias - lr * self.db
    
    def train(self, x, y, lr=3e-3, epochs=1000):
        y = np.where(y <= 0, -1., 1.)
        for e in tqdm(range(epochs)):
            for idx, x_i in enumerate(x):
                self.backward(x_i, y[idx])
                self.update(lr)