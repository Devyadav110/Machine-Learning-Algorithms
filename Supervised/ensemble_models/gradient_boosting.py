from shared import np
from Supervised.ensemble_models.decision_tree import Decision_Tree

# @title **Gradient Boosting Implementation**

class Gradient_Boosting_Regressor():
    """Gradient Boosting Regressor for continuous target variables using L2 Loss."""

    def __init__(self, n_estimators=2, learning_rate=0.1, max_depth=5, min_samples_split=5, min_samples_leaf=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.shape[0] != len(y):
            raise ValueError("The number of observations in X and y must be the same.")

        self.trees = []

        # Initialize predictions with the mean of target values (minimizes L2 loss initially)
        self.initial_prediction = np.mean(y)
        current_prediction = np.full(len(y), self.initial_prediction)

        for m in range(self.n_estimators):
            # Compute pseudo-residuals: the negative gradient of the L2 loss function
            negative_gradient = y - current_prediction

            # Fit a regression tree to the residuals
            model = Decision_Tree(
                purpose="regression",
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )

            model.fit(X, negative_gradient)
            prediction = model.predict(X)

            # Update current predictions by moving in the direction of the negative gradient
            current_prediction += self.learning_rate * prediction
            self.trees.append(model)

        return self

    def predict(self, X):
        """Predict continuous values for X."""
        X = np.asarray(X)
        # Start with the initial constant prediction
        predictions = np.full(X.shape[0], self.initial_prediction)

        # Add the weighted contributions from each sequential tree
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        return predictions


class Gradient_boosting_classification():
    """Gradient Boosting for binary classification (0 and 1) using Log-Loss."""

    def __init__(self, threshold=0.5, n_estimators=2, learning_rate=0.1, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.trees = []
        self.leaf_weights = []
        self.initial_score = None

    def _sigmoid(self, x):
        """Logistic sigmoid function to map scores to probabilities."""
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if not np.array_equal(np.unique(y), np.array([0, 1])):
            raise ValueError("Target y must contain exactly the classes 0 and 1.")

        self.trees = []
        self.leaf_weights = []

        # Step 1: Initialize model with a constant value (log-odds)
        pi = np.sum(y) / len(y)
        self.initial_score = np.log(pi) - np.log(1 - pi)
        current_score = np.full(len(y), self.initial_score)

        for iteration in range(self.n_estimators):
            # Transform log-odds scores to probabilities
            probabilities = self._sigmoid(current_score)

            # Step 2: Calculate residuals (negative gradient of Log-Loss)
            negative_gradient = y - probabilities

            # Step 3: Fit a regression tree to the residuals
            model = Decision_Tree(
                purpose="regression",
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            model.fit(X, negative_gradient)

            # Step 4: Calculate optimal terminal node values (leaf weights)
            # We group samples by the leaf they fall into to update the leaf weight
            model_leaf_weights = {}
            for idx in range(X.shape[0]):
                leaf_id = model._get_leaf_id(X[idx, :])
                if leaf_id not in model_leaf_weights:
                    model_leaf_weights[leaf_id] = [idx]
                else:
                    model_leaf_weights[leaf_id].append(idx)

            # Use Newton-Raphson step to update leaf values for log-loss
            for leaf_id, sample_indices in model_leaf_weights.items():
                p_slice = probabilities[sample_indices]
                g = np.sum(negative_gradient[sample_indices])
                h = np.sum(p_slice * (1 - p_slice))

                # Weight update: sum(residuals) / sum(p * (1-p))
                step = g / h if h != 0 else 0
                current_score[sample_indices] += self.learning_rate * step
                model_leaf_weights[leaf_id] = step

            self.trees.append(model)
            self.leaf_weights.append(model_leaf_weights)

        return self

    def _predict_proba(self, x):
        """Helper to calculate probability for a single sample."""
        x = np.asarray(x)
        score = self.initial_score

        for tree_idx in range(len(self.trees)):
            leaf_id = self.trees[tree_idx]._get_leaf_id(x)
            # Add contribution from each tree's specific leaf node
            score += self.learning_rate * self.leaf_weights[tree_idx].get(leaf_id, 0)

        return self._sigmoid(score)

    def predict(self, X):
        """Predict binary class labels for X based on the probability threshold."""
        X = np.asarray(X)
        # Using list comprehension for prediction is optimized for custom tree traversal
        prediction_proba = np.array([self._predict_proba(x) for x in X])

        return (prediction_proba >= self.threshold).astype(int)

