from shared import np

class IsolationTree() :
    """
    An individual Isolation Tree (iTree) used inside an Isolation Forest.
    It recursively splits data by choosing a random feature and a random split point
    between the minimum and maximum values of that feature in the current dataset.
    """
    class Node() :
        """
        Represents a single node in the Isolation Tree.
        Can be an internal decision node or a terminal leaf node.
        """
        def __init__(
            self,
            *,
            is_leaf = None,
            feature = None,
            threshold = None,
            left_child = None,
            right_child = None,
            size = None
        ) :
            self.is_leaf = is_leaf      # True if this is a leaf node, False/None otherwise
            self.feature = feature      # Index of the feature used for splitting (for decision nodes)
            self.threshold = threshold  # Threshold value for splitting (for decision nodes)
            self.left_child = left_child
            self.right_child = right_child
            self.size = size            # Number of training samples remaining in this node

    def __init__(self, max_depth, seed) :
        """
        Initialize the tree with a maximum depth constraint and a random seed.
        """
        self.max_depth = max_depth
        self.rng = np.random.default_rng(seed=seed)
        self.root_ = None

    def split(self, data, features) :
        """
        Randomly selects a feature from the given list of features, and randomly
        chooses a threshold uniformly distributed between the min and max values of that feature.
        """
        data = np.asarray(data)

        # Select a random feature index from available splittable features
        feature = int(self.rng.choice(features))
        x = data[:, feature]
        max_val, min_val = np.max(x), np.min(x)

        # Uniformly sample a threshold value between the min and max values
        threshold = self.rng.uniform(min_val, max_val)

        return feature, threshold

    def fit(self, X) :
        """
        Builds the isolation tree from the input dataset X.
        """
        X = np.asarray(X)

        self.root_ = self.build_tree(
            X,
            depth = 0
        )

    def build_tree(self, X, depth) :
        """
        Recursively constructs the tree by splitting the data until a stopping criterion is met.
        """
        X = np.asarray(X)
        size = X.shape[0]

        # Stop splitting if max depth is reached
        if depth >= self.max_depth :
            return self.Node(is_leaf = True, size = size)

        # Stop splitting if there is at most 1 element left
        if size <= 1 :
            return self.Node(is_leaf = True, size = size)

        # Find which features have non-zero range (are splittable)
        feature_ranges = np.ptp(X, axis =0)
        splittable_feature = np.flatnonzero(feature_ranges)

        # Stop splitting if all features are constant
        if len(splittable_feature) == 0 :
            return self.Node(is_leaf = True, size = size)

        # Perform the random split
        feature, threshold = self.split(X, splittable_feature)

        # Partition the dataset according to the split criterion
        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask

        # Recursively construct the left and right subtrees
        left_tree = self.build_tree(
            X[left_mask],
            depth = depth + 1
        )

        right_tree = self.build_tree(
            X[right_mask],
            depth = depth + 1
        )

        # Construct and return the decision node
        node = self.Node(
            feature = feature,
            threshold = threshold,
            right_child = right_tree,
            left_child = left_tree
        )

        return node

    def c_factor(self, n):
        """
        Computes the average path length of an unsuccessful search in a Binary Search Tree (BST)
        built on 'n' nodes. Used to normalize the path lengths of sample points.
        """
        if n <= 1:
            return 0.0

        if n == 2:
            return 1.0

        # Euler-Mascheroni constant approximation using the sum of the harmonic series
        harmonic = np.sum(1.0 / np.arange(1, n, dtype=float))

        return 2.0 * harmonic - 2.0 * (n - 1) / n

    def path_length(self, x) :
        """
        Calculates the path length for each sample in 'x' traversing down the isolation tree.
        """
        x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.ndim != 2:
            raise ValueError("X must be a 1D observation or a 2D array.")

        n = x.shape[0]
        pathlength = np.zeros(n)

        for obs in range(n) :
            node = self.root_
            depth = 0

            # Traverse down the tree until reaching a leaf node
            while node is not None and not node.is_leaf :
                if x[obs, node.feature] < node.threshold :
                    node = node.left_child
                else :
                    node = node.right_child

                depth += 1

            # Apply leaf size correction adjustment factors to the final path length
            if node is None:
                pathlength[obs] = depth
            elif node.size is None:
                pathlength[obs] = depth
            elif node.size <= 1 :
                pathlength[obs] = depth
            elif node.size == 2 :
                pathlength[obs] = depth + 1
            else :
                # Leaf node size adjustment using the BST c-factor formula
                pathlength[obs] = depth + self.c_factor(node.size)

        return pathlength

class IsolationForest() :
    """
    An Isolation Forest ensemble for anomaly detection.
    It builds an ensemble of Isolation Trees (iTrees) to isolate observations.
    """
    def __init__(
        self,
        n_estimators = 100,
        max_depth = None,
        max_samples = 256,
        bootstrap = False,
        seed = None
    ) :
        # Validate that we have at least 1 estimator to form an ensemble
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")

        # Validate that max_samples is at least 2 (needed to compute a split)
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2.")

        # Validate that max_depth is not negative
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth cannot be negative.")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.rng = np.random.default_rng(seed=seed)

        self.trees = []

    def fit(self, X) :
        """
        Fit the Isolation Forest ensemble on the training dataset X.
        """
        X = np.asarray(X)

        # Ensure the input dataset is 2-dimensional
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")

        # Ensure the dataset is not empty
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Empty dataset provided for fitting.")

        self.trees = []

        n_samples = X.shape[0]
        # The sample size for each tree is capped by the dataset size
        sample_size = min(self.max_samples, n_samples)

        for _ in range(self.n_estimators):
            # Subsample the dataset for the current tree
            sample_indices = self.rng.choice(
                n_samples,
                size=sample_size,
                replace=self.bootstrap
            )

            # Generate a random seed for the tree's random splits
            seed = self.rng.integers(n_samples)
            X_sample = X[sample_indices]

            # If max_depth is not specified, default to ceil(log2(sample_size))
            if self.max_depth is None :
                max_depth = int(np.ceil(np.log2(sample_size)))
            else :
                max_depth = self.max_depth

            # Instantiate and fit the individual tree
            tree = IsolationTree(max_depth, seed)
            tree.fit(X_sample)

            self.trees.append(tree)

        return self

    def anomaly_score(self, X) :
        """
        Compute the anomaly score for each sample in X.
        The score is bounded between 0 and 1, where higher scores represent higher anomaly probability.
        """
        # Ensure the ensemble has been fitted first
        if len(self.trees) == 0:
            raise RuntimeError("This IsolationForest instance is not fitted yet. Call 'fit' before using anomaly_score.")

        X = np.asarray(X)

        # Ensure the dataset to score is 2-dimensional
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")

        n_samples = X.shape[0]
        if n_samples == 0:
            return np.array([])

        pathlength = np.zeros(n_samples)
        correction = None

        # Aggregate the path lengths from all individual isolation trees
        for tree in self.trees :
            length = tree.path_length(X)
            if correction is None :
                # Compute the average path length of an unsuccessful BST search to normalize scores
                correction = tree.c_factor(min(self.max_samples, n_samples))

            pathlength += length

        # Average the path lengths
        pathlength /= len(self.trees)

        # If correction is 0 (e.g. sample size <= 1), we avoid division by zero
        if correction == 0:
            return np.zeros(n_samples)

        # Apply the standard exponential anomaly score formula: s(x, n) = 2^(-E(h(x)) / c(n))
        scores = np.exp2(-pathlength/correction)

        return scores
