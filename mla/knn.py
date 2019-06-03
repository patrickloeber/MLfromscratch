import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
        """ Calculates the l2 distance between two vectors """
        return np.sqrt(np.sum((x1 - x2)**2, axis = 0))


class KNN:

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None


    def fit(self, X, y):
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)


    def _predict(self, X):
            # Compute distances between x and all examples in the training set. 
            # Sort by class and return only k neighbors
            idx = np.argsort([euclidean_distance(X, x) for x in self.X_train])[:self.k]
            # Extract the labels of the K nearest neighboring training samples
            knn_targets = np.array([self.y_train[i] for i in idx])
            
            # Label sample as the most common class label
            return self._vote(knn_targets)


    def _vote(self, neighbor_labels):
        """ Return the most common class among the neighbor samples """
        most_common = Counter(neighbor_labels).most_common(1)
        return most_common[0][0]
