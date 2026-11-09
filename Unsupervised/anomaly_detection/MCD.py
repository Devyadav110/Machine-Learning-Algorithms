import numpy as np
from scipy.spatial.distance import cdist

# @title **MCD**
class MCD() :
    def __init__(self, reg=0.0, h=0, type='MCD', max_iters=50) :

        if type not in ['MCD', 'Standard'] :
            raise ValueError("The value of the type parameter is wrong")
        else :
            self.type = type

        self.reg = reg
        self.h = h
        self.max_iters = max_iters
        self.determinant_history = None
        self.support_index = None
        self.location = None

    def distance(self, X_train, X_test):
        x_train = np.asarray(X_train)
        x_test = np.asarray(X_test)

        global_mean = np.mean(x_train, axis=0, keepdims=True)

        x_train_centered = x_train - global_mean
        covariance = (
            np.cov(x_train_centered, rowvar=False)
            + self.reg * np.eye(x_train.shape[1])
        )

        assert covariance.shape == (x_train.shape[1], x_train.shape[1])

        x_test_centered = x_test - global_mean
        linear = np.linalg.solve(covariance, x_test_centered.T)

        # Correct Mahalanobis distance calculation: compute only the diagonal elements
        distances_ = np.sum(x_test_centered * linear.T, axis=1)

        return distances_, covariance, global_mean

    def MCD_step(self, X) :
        X = np.asarray(X)

        rng = np.random.default_rng(seed=42)
        index = rng.choice(X.shape[0], size=self.h, replace=False)

        self.determinant_history = []
        prev_distance = np.full(X.shape[0], np.inf, dtype=np.float16)

        for iter in range(self.max_iters) :
            distance_, covariance, location = self.distance(X[index, :], X)

            sign, logdet = np.linalg.slogdet(covariance)
            self.determinant_history.append(sign * logdet)

            if np.allclose(prev_distance, distance_) :
                break

            prev_distance = distance_
            sort_idx = np.argsort(distance_)
            index = sort_idx[:self.h]

        return index, location
