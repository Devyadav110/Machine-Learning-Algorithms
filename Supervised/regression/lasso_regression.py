class custom_lasso_regression() :
    """Lasso Regression using Coordinate Descent."""
    def __init__(self, alpha=0.1, max_iter=1000, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.theta = None

    def _soft_threshold(self, rho, lam):
        if rho < -lam:
            return rho + lam
        elif rho > lam:
            return rho - lam
        else:
            return 0

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        m, n = X.shape
        self.theta = np.zeros(n)
        for _ in range(self.max_iter):
            for j in range(n):
                prediction = X @ self.theta
                rho = X[:, j] @ (y - prediction + self.theta[j] * X[:, j])
                if j == 0 and self.fit_intercept:
                    self.theta[j] = rho / m
                else:
                    self.theta[j] = self._soft_threshold(rho, self.alpha * m) / (X[:, j] @ X[:, j])
        return self

    def predict(self, X):
        X = np.asarray(X)
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.theta
