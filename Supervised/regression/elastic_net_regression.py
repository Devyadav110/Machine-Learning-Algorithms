class custom_elastic_net_regression() :
    """Elastic Net Regression using Coordinate Descent."""
    def __init__(self, lambda_1=0.01, lambda_2=0.01, max_iter=1000):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.max_iter = max_iter
        self.beta = None

    def _soft_threshold(self, rho, lam):
        if rho < -lam:
            return rho + lam
        elif rho > lam:
            return rho - lam
        else:
            return 0

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)

        beta = np.zeros(X.shape[1])
        for _ in range(self.max_iter):
            for j in range(X.shape[1]):
                prediction = X @ beta
                rho = X[:, j] @ (Y - prediction + beta[j] * X[:, j])
                if j == 0:
                    beta[j] = rho / X.shape[0]
                else:
                    beta[j] = self._soft_threshold(rho, self.lambda_1 * X.shape[0]) / (X[:, j] @ X[:, j] + self.lambda_2)
        self.beta = beta
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X @ self.beta
