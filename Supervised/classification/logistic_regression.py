
class custom_logistic_regression() :
    def __init__(self, threshold=0.5, learning_rate=0.01, epochs=100):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta = None

    def _sigmoid(self, x):
        # Prevent overflow in np.exp by using two-way formulation
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.beta = np.zeros(X.shape[1])

        for _ in range(self.epochs):
            prediction = self._sigmoid(X @ self.beta)
            gradient = X.T @ (y - prediction)
            self.beta += self.learning_rate * gradient

        return self

    def predict(self, X):
        X = np.asarray(X)
        proba = self._sigmoid(X @ self.beta)
        return np.where(proba < self.threshold, 0, 1)
