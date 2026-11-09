from shared import np

# @title **Helper function for decision tree**

def variance_split(Y_left, Y_right):
    n = len(Y_left) + len(Y_right)
    if n == 0:
        return 0.0
    else:
        return (len(Y_left) / n) * np.var(Y_left) + (len(Y_right) / n) * np.var(Y_right)

def gini_impurity(Y):
    """Calculates the Gini impurity of a given set of target values."""
    y = np.asarray(Y)
    values, count = np.unique(y, return_counts=True)
    probability = count / len(y)
    return 1 - np.sum(probability ** 2)

def gini_split(y_left, y_right):
    """Calculates the weighted Gini impurity for a binary split."""
    n = len(y_left) + len(y_right)
    if n == 0:
        return 0.0
    return (len(y_left) / n) * gini_impurity(y_left) + (len(y_right) / n) * gini_impurity(y_right)

def candidate_thresholds(X):
    """Generates candidate thresholds for splitting a feature."""
    feature_values = np.sort(np.unique(X))
    return (feature_values[:-1] + feature_values[1:]) / 2

def find_best_split(X, y, min_samples_leaf=1, max_features=None, purpose="classification"):
    """Finds the best feature and threshold to split the data based on Gini impurity."""
    X = np.asarray(X)
    Y = np.asarray(y)

    best_feature, best_threshold, best_gini = None, None, float('inf')

    # ----- Handles no max_featuure being passes -------
    if max_features is None:
        feature_idx = range(X.shape[1])
    else:
        if max_features > X.shape[1]:
            raise ValueError("parameter max_feature is greater than then total number of feature available")
        feature_idx = np.random.choice(range(0, X.shape[1]), max_features, replace=False)

    for feature in feature_idx:
        thresholds = candidate_thresholds(X[:, feature])

        for threshold in thresholds:
            left_mask = X[:, feature] < threshold
            right_mask = ~left_mask

            # Skip invalid splits
            if not left_mask.any() or not right_mask.any():
                continue
            if (left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf):
                continue

            if purpose == "regression":
                gini = variance_split(Y[left_mask], Y[right_mask])
            else:
                gini = gini_split(Y[left_mask], Y[right_mask])

            if gini < best_gini:
                best_feature = feature
                best_threshold = threshold
                best_gini = gini

    return best_feature, best_threshold, best_gini

def leaf_prediction(Y, purpose):
    """Returns the majority class in a given set of target values."""
    if purpose == "regression":
        return np.mean(Y)
    else:
        value, count = np.unique(Y, return_counts=True)
        return value[np.argmax(count)]

class Decision_Tree():
    class Node():
        """Node class for the Decision Tree. Represents a decision point or a leaf in the tree."""
        def __init__(
            self,
            feature=None,  # Index of the feature used for splitting
            threshold=None,
            left_child=None,
            right_child=None,
            *,
            prediction=None,  # Prediction value if this node is a leaf node
            leaf_id=None  # Unique leaf Id if it is a leaf node
        ):
            self.feature = feature
            self.threshold = threshold
            self.left_child = left_child
            self.right_child = right_child
            self.prediction = prediction
            self.leaf_id = leaf_id

    def __init__(self, purpose="classification", max_depth=None, min_samples_split=4, min_samples_leaf=2, min_impurity_decrease=0.0, max_features=None):
        # Initialize hyperparameters for tree construction
        self.purpose = purpose
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features

        self.root = None  # Root node of the decision tree
        self.leaf_count = 0

    def fit(self, X, Y):
        # Convert input to numpy arrays and build the tree
        X = np.asarray(X)
        y = np.asarray(Y)

        self.leaf_count = 0

        self.root = self._build_tree(
            X,
            y,
            depth=0,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features
        )

        return self

    def _build_tree(
        self,
        X,
        y,
        depth,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_impurity_decrease,
        max_features
    ):
        # Determine the prediction for the current node based on the majority class
        prediction = leaf_prediction(y, self.purpose)

        # ---------- Stopping conditions ----------

        # 1. Stop if max depth reached or node is pure
        if max_depth is not None and depth >= max_depth:
            self.leaf_count += 1
            return self.Node(prediction=prediction, leaf_id=self.leaf_count)

        # 2. if the Node is already pure
        if len(np.unique(y)) == 1:  # This also covers pure nodes for regression (variance=0)
            self.leaf_count += 1
            return self.Node(prediction=prediction, leaf_id=self.leaf_count)

        # 3. Stop if not enough samples to split
        if len(y) < min_samples_split:
            self.leaf_count += 1
            return self.Node(prediction=prediction, leaf_id=self.leaf_count)

        # ------ Find the best feature and threshold to split the data --------
        feature, threshold, impurity = find_best_split(X, y, min_samples_leaf, max_features, self.purpose)

        # 4. Stop if no valid split exists
        if feature is None:
            self.leaf_count += 1
            return self.Node(prediction=prediction, leaf_id=self.leaf_count)

        # -------- Minimum impurity reduction --------
        if self.purpose == "regression":
            parent_impurity = np.var(y)
        else:
            parent_impurity = gini_impurity(y)

        # Stop if impurity reduction is below threshold
        if parent_impurity - impurity <= min_impurity_decrease:
            self.leaf_count += 1
            return self.Node(prediction=prediction, leaf_id=self.leaf_count)

        # Create masks for splitting the data into left and right children
        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask

        # Recursively build the left and right child trees
        left_child = self._build_tree(
            X[left_mask],
            y[left_mask],
            depth + 1,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            max_features
        )
        right_child = self._build_tree(
            X[right_mask],
            y[right_mask],
            depth + 1,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            min_impurity_decrease,
            max_features
        )

        # Return a new Node representing the current split
        return self.Node(
            feature=feature,
            threshold=threshold,
            left_child=left_child,
            right_child=right_child
        )

    def _get_leaf_id(self, x):
        x = np.asarray(x)
        node = self.root

        while node.leaf_id is None:  # Traverse until a leaf node
            if x[node.feature] < node.threshold:
                node = node.left_child
            else:
                node = node.right_child

        return node.leaf_id

    def _predict_one(self, x):
        # Traverse the tree for a single prediction
        x = np.asarray(x)
        node = self.root

        while node.prediction is None:
            if x[node.feature] < node.threshold:
                node = node.left_child
            else:
                node = node.right_child

        return node.prediction

    def predict(self, X):
        # Predict for an array of samples
        X = np.asarray(X)

        if self.root is None:
            raise RuntimeError("Tree has not been fitted yet")

        # Handle scalar input (0-dimensional array)
        if X.ndim == 0:
            X = X.reshape(1, 1)  # Treat as 1 sample, 1 feature

        # Handle 1D input (single sample with multiple features)
        elif X.ndim == 1:
            X = X.reshape(1, -1)  # Treat as 1 sample, N features

        return np.array([self._predict_one(x) for x in X])

