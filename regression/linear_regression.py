import tqdm
import numpy as np

# Linear regression
class LinReg:

    # Constructor
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.random.randn(num_features, 1)
        self.b = np.random.randn()
    
    # Feed Forward
    def forward(self, X):
        y = np.dot(X, self.W) + self.b
        return y

    # Loss function
    def compute_loss(self, y, y_true):
        loss = np.sum(np.square(y - y_true))
        return loss/(2*y.shape[0])
    
    # Backward pass
    def backward(self, X, y_true, y_hat):
        m = y_hat.shape[0]
        db = np.sum(y_hat - y_true)/m
        dW = np.sum(np.dot(np.transpose(y_hat - y_true), X), axis=0)/m
        return dW, db
    
    # Updating parameters
    def update_params(self, dW, db, lr):
        self.W = self.W - lr * np.reshape(dW, (self.num_features, 1))
        self.b = self.b - lr * db
    
    # Training
    def train(self, x_train, y_train, iterations, lr):
        losses = []
        with tqdm.trange(iterations) as t:
            for _ in t:
                y_hat = self.forward(x_train)
                dW, db = self.backward(x_train, y_train, y_hat)
                self.update_params(dW, db, lr)
                loss = self.compute_loss(y_hat, y_train)
                losses.append(loss)
                t.set_description(f'Loss: {loss}')
        return losses

# Data
x = np.arange(0,10,0.01).reshape(-1, 1)
y = [5*i+10 for i in x]

# Checking
model = LinReg(1)
losses = model.train(x, y, 10000, 3e-3)
print(model.W, model.b)