import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from logistic_regression import LogReg

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state=1234)

num_features = X.shape[1]
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

model = LogReg(num_features)
losses = model.train(x_train, y_train, 10000, 3e-6)
print(model.accuracy(x_test, y_test))