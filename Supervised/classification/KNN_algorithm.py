class custom_KNN_algorithm() :
    def __init__(self, K=2, p=2):
        self.K = K
        self.x_t = None
        self.y_t = None
        self.p = p

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.x_t = X
        self.y_t = y
        return self

    def predict(self, X):
        X = np.asarray(X)
        predictions = []

        for x in X:
            # Calculate minkowski distance to every training point
            distances = np.sum(abs((self.x_t - x)) ** self.p, axis=1)

            # Indices of K nearest points
            k_indices = np.argsort(distances)[:self.K]
            # Their labels
            k_labels = self.y_t[k_indices]
            # Majority vote
            labels, counts = np.unique(k_labels, return_counts=True)
            prediction = labels[np.argmax(counts)]

            predictions.append(prediction)

        return np.array(predictions)

