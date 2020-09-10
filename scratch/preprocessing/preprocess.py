import random
import numpy as np

def split_data(data, prob):
    data = data[:]
    random.shuffle(data)
    cut = ont(len(data) * prob)
    return data[:cut], data[cut:]

def train_test_split(x, y, prob):
    idx = [i for i in range(len(x))]
    train_idx, test_idx = split_data(idx, (1 - prob))
    return ([x[i] for i in train_idx],
            [x[i] for i in test_idx],
            [y[i] for i in train_idx],
            [y[i] for i in test_idx])

def normalize(x, axis=0):
    mu = np.mean(x, axis=axis)
    sigma = np.std(x, axis=axis)
    return (x - mu) / sigma

def denormalize(x_org, x_norm, axis=0):
    mu = np.mean(x_org, axis=axis)
    sigma = np.std(x_org, axis=axis)
    return (x_norm * sigma) + mu