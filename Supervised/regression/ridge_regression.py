class custom_ridge_regression() :
    """Ridge Regression supporting both Gradient Descent and Normal Equation."""
    def __init__(self, alpha=0.1, method='Gradient', fit_intercept=True, learning_rate=0.01, epochs=100):
        self.alpha = alpha
        self.method = method
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None

    def fit(self, X, Y):
        x, y = np.asarray(X), np.asarray(Y)
        if self.fit_intercept:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        self.theta = np.zeros(x.shape[1])
        if self.method == 'Gradient':
            for _ in range(self.epochs):
                prediction = x @ self.theta
                error = y - prediction
                gradient = (2/x.shape[0]) * (x.T @ error) + self.alpha * self.theta
                self.theta += self.learning_rate * gradient
        else:
            self.theta = np.linalg.inv(x.T @ x + self.alpha * np.identity(x.shape[1])) @ x.T @ y
        return self

    def predict(self, X):
        x = np.asarray(X)
        if self.fit_intercept:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x @ self.theta
