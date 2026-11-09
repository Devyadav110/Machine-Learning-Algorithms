from shared import np

class Support_Vector_Classifier() :
  def __init__(self, C, tol = 0.1, passes = 2, kernel = "linear", max_passes = 100, verbose=False) :
    """
    Initializes the Support Vector Classifier.

    Args:
      C (float): Regularization parameter. The strength of the regularization is inversely proportional to C.
      tol (float): Tolerance for the KKT conditions (stopping criterion).
      passes (int): Maximum number of times to iterate over the entire dataset without any alpha changes.
      kernel (str): Type of kernel function to use. Currently, only 'linear' is supported.
    """
    self.C = C
    self.tol = tol
    self.max_passes = max_passes
    self.kernel = kernel
    self.alpha_ = None
    self.alpha = None # Lagrange multipliers
    self.bias = None # Bias term
    self.X_sv = None # Support vectors (features)
    self.y_sv = None # Support vectors (labels)
    self.weights = None # Weights for the support vectors
    self.error_cache_ = None

  def fit(self, X, Y) :
    X = np.asarray(X, dtype=float)
    y = np.asarray(Y, dtype=float)

    # Initialize Lagrange multipliers (alpha), bias, and error cache
    alpha = np.zeros(len(y))
    self.bias = 0.0
    error_cache = y * -1.0

    # Reset support vectors and alpha for new training
    self.X_sv = None
    self.y_sv = None
    self.alpha = None

    total_passes = 0

    # Main SMO loop
    while total_passes <= self.max_passes :
      total_passes += 1

      for i_idx in range(X.shape[0]) :
        # Check KKT conditions for i_idx
        kkt_violation_1 = (alpha[i_idx] < self.C) and (y[i_idx] * error_cache[i_idx] + self.tol < 0)
        kkt_violation_2 = (alpha[i_idx] > 0) and (y[i_idx] * error_cache[i_idx] - self.tol > 0)

        if kkt_violation_1 or kkt_violation_2 :
          j_idx, max_diff = None, float('-inf')

          # Second choice heuristic: Find j_idx that maximizes |E_i - E_j|
          candidate_js = np.argsort(-np.abs(error_cache[i_idx] - error_cache))

          pass_updated = False

          for j_idx in candidate_js :
            if i_idx == j_idx :
              continue

            eta = np.sum((X[i_idx] - X[j_idx])** 2)
            if eta <= 1e-12 :
              continue

            alpha_j_unc = alpha[j_idx] + (y[j_idx] * (error_cache[i_idx] - error_cache[j_idx])) / eta

            if y[i_idx] != y[j_idx] :
              L = max(0.0, alpha[j_idx] - alpha[i_idx])
              H = min(self.C, self.C + alpha[j_idx] - alpha[i_idx])
            else :
              L = max(0.0, alpha[i_idx] + alpha[j_idx] - self.C)
              H = min(self.C, alpha[i_idx] + alpha[j_idx])

            # Clip alpha_j_new within the bounds [L, H]
            alpha_j_new = np.clip(alpha_j_unc, L, H)

            if abs(alpha_j_new - alpha[j_idx]) < 1e-5:
              continue

            pass_updated = True

            # Update alpha_i based on the new alpha_j
            alpha_i_new = alpha[i_idx] + (y[i_idx] * y[j_idx]) * (alpha[j_idx] - alpha_j_new)

            # Calculate the change in alpha values
            delta_alpha_i = alpha_i_new - alpha[i_idx]
            delta_alpha_j = alpha_j_new - alpha[j_idx]

            # Update bias (b)
            b_1 = (self.bias
                  - error_cache[i_idx]
                  - y[i_idx] * delta_alpha_i * (X[i_idx] @ X[i_idx])
                  - y[j_idx] * delta_alpha_j * (X[i_idx] @ X[j_idx]))

            b_2 = (self.bias
                  - error_cache[j_idx]
                  - y[j_idx] * delta_alpha_j * (X[j_idx] @ X[j_idx])
                  - y[i_idx] * delta_alpha_i * (X[i_idx] @ X[j_idx]))

            # Choose the new bias b_new
            if alpha_i_new > 0 and alpha_i_new < self.C :
              b_new = b_1
            elif alpha_j_new > 0 and alpha_j_new < self.C :
              b_new = b_2
            else :
              b_new = (b_1 + b_2) / 2

            delta_b = b_new - self.bias

            error_cache += (
                y[i_idx] * delta_alpha_i * (X @ X[i_idx])
                + y[j_idx] * delta_alpha_j * (X @ X[j_idx])
                + delta_b
            )

            old_objective = self.dual_objective(alpha, y, X)

            # Apply the updates to alpha and bias
            alpha[i_idx] = alpha_i_new
            alpha[j_idx] = alpha_j_new
            self.bias = b_new
            new_objective = self.dual_objective(alpha, y, X)

            if new_objective < old_objective - 1e-8:
                print(f"WARNING: dual objective decreased : {old_objective} -> {new_objective}")
            break

      violation = (
          ((alpha < self.C) & (y * error_cache < -self.tol))
          |
          ((alpha > 0) & (y * error_cache > self.tol))
      )

      num_violation = np.sum(violation)

      if num_violation == 0 :
        break

    # Identify support vectors (where alpha > 0)
    mask = alpha > 1e-5

    self.alpha = alpha[mask]
    self.alpha_ = alpha
    self.X_sv = X[mask]
    self.y_sv = y[mask]
    self.error_cache_ = error_cache

    self.weights = np.sum((self.alpha * self.y_sv)[:, np.newaxis] * self.X_sv, axis=0)

    return self

  def dual_objective(self, alpha, y, X):
    K = X @ X.T
    ay = alpha * y

    return (
        np.sum(alpha)
        - 0.5 * ay @ K @ ay
    )

  def decision_function(self, X) :
    X = np.asarray(X, dtype=float)

    if X.ndim == 1 :
      X = X[np.newaxis, :]

    kernel_matrix =  X @ self.X_sv.T
    decision = kernel_matrix @ (self.alpha * self.y_sv) + self.bias

    return decision

  def predict(self, X) :
    X = np.asarray(X, dtype=float)

    decision = self.decision_function(X)
    predictions = np.where(decision > 0, 1, -1)

    return predictions
