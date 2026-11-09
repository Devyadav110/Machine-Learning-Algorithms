### CUSTOM REGRESSION CLASSES ###

class custom_linear_regression() :
    """Linear Regression using the Normal Equation (OLS)."""
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.theta = None

    def fit(self, X, Y):
        x_num = np.asarray(X)
        y_num = np.asarray(Y)
        if self.fit_intercept:
            x_num = np.hstack((np.ones((x_num.shape[0], 1)), x_num))
        self.theta = np.linalg.inv(x_num.T @ x_num) @ x_num.T @ y_num
        return self

    def predict(self, X):
        X_num = np.asarray(X)
        if self.fit_intercept:
            X_num = np.hstack((np.ones((X_num.shape[0], 1)), X_num))
        return X_num @ self.theta

class custom_gradient_descent() :
    """Linear Regression using Batch Gradient Descent."""
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None

    def fit(self, X, Y):
        x = np.asarray(X)
        y = np.asarray(Y)
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        self.theta = np.zeros(x.shape[1])
        for _ in range(self.epochs):
            prediction = x @ self.theta
            error = y - prediction
            gradient = (2/x.shape[0]) * (x.T @ error)
            self.theta += self.learning_rate * gradient
        return self

    def predict(self, X):
        x = np.asarray(X)
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x @ self.theta
