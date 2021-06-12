import numpy as np


class BaseRegression:
    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000):
        # Assign the variables
        self.learning_rate = learning_rate
        self.n_iters = n_iters

        # Weights and bias
        self.weights, self.bias = None, None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights, self.bias = np.zeros(n_features), 0

        # Minimizing loss, and finding the correct Weights and biases using Gradient Descent
        for _ in range(self.n_iters):
            y_predicted = self._approximation(X, self.weights, self.bias)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return self._predict(X, self.weights, self.bias)

    def _predict(self, X, w, b):
        raise NotImplementedError

    def _approximation(self, X, w, b):
        raise NotImplementedError


class LinearRegression(BaseRegression):
    def _approximation(self, X, w, b):
        return np.dot(X, w) + b

    def _predict(self, X, w, b):
        return np.dot(X, w) + b


class LogisticRegression(BaseRegression):
    def _approximation(self, X, w, b):
        linear_model = np.dot(X, w) + b
        return self._sigmoid(linear_model)

    def _predict(self, X, w, b):
        linear_model = np.dot(X, w) + b
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (np.exp(-x) + 1)
