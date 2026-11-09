class Linear_Discriminant_Analysis:
    """
    Linear Discriminant Analysis (LDA) classifier and dimensionality reduction technique.
    """
    def __init__(self, n_components=None, reg=0.0):
        """
        Initializes the LDA model.

        Parameters:
        - n_components (int): Number of components for dimensionality reduction.
        - reg (float): Regularization constant for the within-class scatter matrix.
        """
        if reg < 0:
            raise ValueError("reg must be non-negative")

        self.n_components = n_components
        self.reg = reg

        # Scatter matrices
        self.S_w = None
        self.S_b = None
        self.S_w_regularized = None

        # Class characteristics
        self.classes_ = None
        self.class_means_ = None
        self.class_count_ = None
        self.mean_ = None

        # Decomposition attributes
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.w_ = None               # Projection weights
        self.components_ = None
        self.n_features_in_ = None
        self.discriminability_ratio_ = None

    def fit(self, X, y):
        """
        Fits the LDA model according to the given training data X and class labels y.
        """
        # --- Input validation and type casting ---
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(
                "X must contain samples and features"
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "X must contain only finite values"
            )

        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        if y.shape[0] != X.shape[0]:
            raise ValueError(
                "X and y must have the same number of samples"
            )

        self.classes_ = np.unique(y, )
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Set default number of components if not provided
        n_classes = len(self.classes_)
        if self.n_components is None:
            self.n_components = min(n_features, n_classes - 1)
        else:
            if not 1 <= self.n_components <= min(n_features, n_classes - 1):
                raise ValueError(
                    f"n_components must be between 1 and min(n_features, n_classes - 1) = {min(n_features, n_classes - 1)}"
                )

            if not isinstance(self.n_components, (int, np.integer)) :
                raise ValueError(
                    "n_components should be an integer"
                )

        if len(self.classes_) < 2 :
            raise ValueError(
                "There must be atleast 2 classes"
            )

        # --- Initialize structures ---
        self.S_w = np.zeros((n_features, n_features))
        self.S_b = np.zeros((n_features, n_features))
        self.class_count_ = []
        self.class_means_ = {}

        # Overall dataset mean
        self.mean_ = np.mean(X, axis=0)[:, np.newaxis]

        # --- Compute Scatter Matrices (Between-class and Within-class) ---
        for cls in self.classes_:
            X_c = X[y == cls]
            n_samples_c = X_c.shape[0]

            # Calculate class mean
            mean_c = np.mean(X_c, axis=0)[:, np.newaxis]
            self.class_means_[int(cls)] = mean_c.ravel()
            self.class_count_.append(n_samples_c)

            # Within-class scatter contribution
            centered_X_c = X_c - mean_c.T
            self.S_w += centered_X_c.T @ centered_X_c

            # Between-class scatter contribution
            mean_diff = mean_c - self.mean_
            self.S_b += n_samples_c * (mean_diff @ mean_diff.T)

        # --- Apply Regularization to Within-class Scatter Matrix ---
        scale = np.trace(self.S_w) / n_features
        if scale <= 0:
            scale = 1.0

        self.S_w_regularized = (
            self.S_w
            + self.reg * scale * np.eye(n_features)
        )

        # --- Solve the Generalized Eigenvalue Problem ---
        try:
            eigenvalues, eigenvectors = eigh(self.S_b, self.S_w_regularized)
        except LinAlgError as exc:
            raise ValueError(
                "The within-class scatter matrix is singular or not positive-definite. "
                "Try setting reg > 0."
            ) from exc

        # Sort eigenvectors in descending order of their eigenvalues
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Keep only the requested number of components
        self.eigenvalues_ = eigenvalues[:self.n_components]
        self.components_ = eigenvectors[:, :self.n_components]
        self.w_ = self.components_  # Assign projection weights

        # --- Calculate Discriminability Ratio (Explained Variance) ---
        positive_values = np.maximum(self.eigenvalues_, 0.0)
        total_variance = positive_values.sum()

        if total_variance > 0:
            self.discriminability_ratio_ = positive_values / total_variance
        else:
            self.discriminability_ratio_ = np.zeros_like(positive_values)

        return self

    def transform(self, X):
        """
        Projects the input features into the low-dimensional LDA space.
        """
        if self.components_ is None:
            raise ValueError("Fit the model before transforming data")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but received {X.shape[1]}"
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        # Linear projection
        return (X - self.mean_.T) @ self.components_

    def fit_transform(self, X, y) :

        return self.fit(X, y).transform(X)

