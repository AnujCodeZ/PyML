import tqdm
import numpy as np

# Logistic Regression
class LogReg:

    # Constructor
    def __init__(self, num_features, num_labels):

        self.weights = np.random.rand(num_features, num_labels) * 0.01
        self.bias = np.zeros((1, num_labels))
        self.num_features = num_features
        self.num_labels = num_labels

    # Activations
    def sigmoid(self, Z):

        return 1. / (1. + np.exp(-Z))

    def softmax(self, Z):

        return np.exp(Z) / np.sum(Z)
    
    # Forward pass
    def forward(self, X):

        Z = np.matmul(X, self.weights) + self.bias
        
        if self.num_labels == 1:
            A = self.sigmoid(Z)
        else:
            A = self.softmax(Z)
        return A
    
    # Cost function
    def compute_cost(self, A, Y):

        loss = -1 * (np.dot(np.transpose(Y), np.log(A)) + \
                           np.dot(np.transpose(1-Y), np.log(1-A)))
        
        loss = loss / Y.shape[0]

        return np.squeeze(loss)
    
    # Backward pass
    def backward(self, X, Y, A):
        m = X.shape[0]
        dZ = A - Y
        dW = np.dot(np.transpose(X), dZ) / m
        db = np.sum(dZ) / m

        return dW, db
    
    # update parameters
    def update_params(self, dW, db, lr):

        self.weights -= lr * dW
        self.bias -= lr * db
    
    # Prediction
    def predict(self, X):

        A = self.forward(X)
        Y_hat = A > 0.5

        return Y_hat
    
    # Accuracy
    def accuracy(self, X, Y):

        Y_hat = self.predict(X)
        accuracy = np.sum(Y_hat == Y) / X.shape[0]

        return accuracy
    # Training.
    def train(self, x_train, y_train, iterations=3000, lr=0.03):
        losses = []
        with tqdm.trange(iterations) as t:
            for _ in t:
                y_hat = self.forward(x_train)
                dW, db = self.backward(x_train, y_train, y_hat)
                self.update_params(dW, db, lr)
                loss = self.compute_cost(y_hat, y_train)
                losses.append(loss)
                t.set_description(f'Loss: {loss}')
        return losses