from shared import np

class PCA:
    """
    Principal Component Analysis (PCA) for dimensionality reduction.
    """
    def __init__(self, n_component_=1, decomposition='EVD'):
        """
        Initializes the PCA model.

        Parameters:
        - n_component_ (int): The number of principal components to retain.
        - decomposition (str): 'EVD' (Eigenvalue Decomposition) or 'SVD' (Singular Value Decomposition).
        """
        if not isinstance(n_component_, int) or n_component_ <= 0:
            raise ValueError("n_component_ must be a positive integer")

        if decomposition not in ['EVD', 'SVD']:
            raise ValueError("decomposition is not supported")

        self.n_component_ = n_component_
        self.decomposition = decomposition

        # Model state variables
        self.mean_ = None
        self.x_centered_ = None
        self.components_ = None
        self.eigenvalues_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

        if self.decomposition == 'EVD':
            self.covariance_matrix_ = None

    def fit(self, X):
        """
        Computes the principal components (eigenvectors) and eigenvalues
        from the training data X using either EVD or SVD.
        """
        # --- Input validation and preprocessing ---
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("The matrix should be of shape (n_samples, n_features)")

        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError("PCA requires at least two observations")

        if not 1 <= self.n_component_ <= min(n_samples, n_features):
            raise ValueError(
                f"n_component_ ({self.n_component_}) must not exceed min(n_samples, n_features) = {min(n_samples, n_features)}"
            )

        # --- Data centering ---
        self.mean_ = np.mean(X, axis=0)
        self.x_centered_ = X - self.mean_

        # --- Decomposition stage ---
        if self.decomposition == 'EVD':
            # Compute covariance matrix
            self.covariance_matrix_ = np.cov(self.x_centered_, rowvar=False)

            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)

            # Sort eigenvalues and eigenvectors in descending order
            sort_idx = np.argsort(eigenvalues)[::-1]
            self.eigenvalues_ = eigenvalues[sort_idx]
            eigenvectors = eigenvectors[:, sort_idx]

            # Select principal components (projection vectors) with shape (n_component_, n_features)
            self.components_ = eigenvectors[:, :self.n_component_].T

        else:
            # Perform SVD on the centered data
            U, S, Vt = np.linalg.svd(self.x_centered_, full_matrices=False)

            # Singular values S are related to eigenvalues of covariance matrix
            self.eigenvalues_ = (S ** 2) / (n_samples - 1)

            # Select principal components from the right singular vectors Vt with shape (n_component_, n_features)
            self.components_ = Vt[:self.n_component_, :]

        # --- Variance calculations ---
        eigen_sum = np.sum(self.eigenvalues_)
        if eigen_sum <= 0:
            raise ValueError(
                "PCA cannot calculate explained variance because "
                "the data has zero total variance"
            )

        self.explained_variance_ = self.eigenvalues_[:self.n_component_]
        self.explained_variance_ratio_ = self.explained_variance_ / eigen_sum

        return self

    def fit_transform(self, X):
        """
        Fits the model with X and applies the dimensionality reduction on X.
        Also calculates the explained variance and explained variance ratio.
        """
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        """
        Projects the data X into the lower-dimensional PCA space.
        """
        # --- Validation of projection input ---
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("Provide a valid 2D dimension data")

        if self.components_ is None:
            raise ValueError("Fit the data first before transforming")

        if X.shape[1] != len(self.mean_):
            raise ValueError("X has an incorrect number of features")

        # --- Linear transformation ---
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def inverse_transform(self, Z):
        """
        Transforms data from the PCA space back to the original space.
        """
        # --- Validation of inverse input ---
        Z = np.asarray(Z, dtype=float)

        if self.components_ is None:
            raise ValueError("Fit the data first before reconstructing")

        if Z.ndim != 2:
            raise ValueError("Z must be a 2D array")

        if Z.shape[1] != self.n_component_:
            raise ValueError("Z has an incorrect number of components")

        # --- Reconstruction logic ---
        return Z @ self.components_ + self.mean_
