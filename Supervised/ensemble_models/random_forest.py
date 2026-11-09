
# @title **Random Forest**
class Random_Forest_Classifier():
    def __init__(self, max_depth=3, min_samples_split=4, min_samples_leaf=2, min_impurity_decrease=0.0, B=10, max_features=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.B = B
        self.max_features = max_features
        self.trees = []
        self.bootstrap_indices = []
        self.oob_indices = []

    def sampling(self, X):
        row_idx = X.shape[0]
        return np.random.randint(0, row_idx, row_idx)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.trees = []
        self.bootstrap_indices = []
        self.oob_indices = []

        for i in range(self.B):
            index = self.sampling(X)
            X_bootstrap, y_bootstrap = X[index], y[index]

            self.bootstrap_indices.append(index)
            self.oob_indices.append(set(np.setdiff1d(range(0, X.shape[0]), index)))

            tree = Decision_Tree(
                purpose="classification",
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features
            )

            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

        return self

    def _predict_one(self, x):
        prediction = []

        for i in range(len(self.trees)):
            prediction.append(self.trees[i]._predict_one(x))

        value, count = np.unique(prediction, return_counts=True)
        return value[np.argmax(count)]

    def predict(self, X):
        X = np.asarray(X)

        if len(self.trees) == 0:
            raise RuntimeError("The classifier has not fitted yet, call fit function and try again.")

        return np.array([self._predict_one(x) for x in X])

    def oob_predict(self, X):
        X = np.asarray(X)
        oob_predictions = []

        for i in range(X.shape[0]):
            predictions = []
            for j in range(len(self.oob_indices)):
                if i in self.oob_indices[j]:
                    predictions.append(self.trees[j].predict(X[i]))
            if len(predictions) != 0:
                value, count = np.unique(predictions, return_counts=True)
                oob_predictions.append(value[np.argmax(count)])
            else:
                oob_predictions.append(None)

        return oob_predictions

    def oob_score(self, X, y):
        oob_predictions_raw = self.oob_predict(X)
        y = np.asarray(y)

        valid_indices = [i for i, pred in enumerate(oob_predictions_raw) if pred is not None]

        if not valid_indices:
            return 0.0

        oob_predictions = np.array([oob_predictions_raw[i] for i in valid_indices])
        true_labels = np.array([y[i] for i in valid_indices])

        total_correct = np.sum(oob_predictions == true_labels)
        return total_correct / len(oob_predictions)

