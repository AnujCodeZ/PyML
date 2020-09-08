import tqdm
import numpy as np
from typing import List, Tuple

Vector = List[float]

def predict(weight: float, bias: float, x_i: float) -> float:
    """Returns the prediction"""
    return weight * x_i + bias

def error(weight: float, bias: float, x_i: float, y_i: float) -> float:
    """
    Returns how different our prediction weight * x_i + bias
    from actual value y_i
    """
    return predict(weight, bias, x_i) - y_i

def sum_sq_error(weight: float, bias: float, x: Vector, y: Vector) -> float:
    """Returns sum of squared errors"""
    return sum(error(weight, bias, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))

np.random.seed(0)

num_epochs = 10000
learning_rate = 1e-5
weight = np.random.random()
bias = np.random.random()
x = np.arange(1, 10, 0.01)
y = [3 * i - 5 for i in x]

with tqdm.trange(num_epochs) as t:
    for _ in t:
        # Partial differenciation of weight wrt loss
        dw = sum(2 * error(weight, bias, x_i, y_i) * x_i
                 for x_i, y_i in zip(x, y))
        # Partial differenciation of bias wrt loss
        db = sum(2 * error(weight, bias, x_i, y_i)
                 for x_i, y_i in zip(x, y))
        # Loss
        loss = sum_sq_error(weight, bias, x, y)
        t.set_description(f'loss: {loss:.3f}')
        # Gradient step
        weight = weight - learning_rate * dw
        bias = bias - learning_rate * db
        
print(weight, bias)