class K_Mean() :
  def __init__(self, k, tol=0.001, max_iters=50, init='random') :
    self.k = k
    self.tol = tol
    self.max_iters = max_iters
    self.init =  init
    self.centroids_ = None
    self.labels_ = None
    self.errors_ = []
    self.n_itr_ = None

  def fit(self, X) :
    X = np.asarray(X)

    # Choose initial centroids
    if isinstance(self.init, np.ndarray) :
      centroids = self.init.copy()
    elif self.init == 'k_mean++' :
      centroids = self._init_centroids(X)
    elif self.init == 'random' :
      indices = np.random.choice(X.shape[0], self.k, replace=False)
      centroids = X[indices].copy()
    else :
      raise ValueError("Invalid initialization method")

    # Initialize variables
    labels = np.zeros(X.shape[0], dtype=np.int8)
    self.n_itr_ = 0

    # Main expectation-maximization loop
    for _ in range(self.max_iters) :
      # Assign points to nearest centroids and compute WCSS error
      new_labels, error = self._WCSS(X, centroids, return_value=['labels', 'error'])
      self.errors_.append(error)

      # Update centroids to be the mean of assigned points
      new_centroids = np.zeros(centroids.shape)
      for ctr in range(self.k) :
        centroid = (new_labels == ctr)
        if X[centroid].shape[0] == 0 :
          continue
        new_centroids[ctr] = np.mean(X[centroid], axis=0)

      # Check for convergence
      centroid_change = np.max(np.abs(new_centroids - centroids))
      labels_unchanged = np.array_equal(new_labels, labels)

      centroids = new_centroids
      labels = new_labels

      self.n_itr_ += 1

      if centroid_change < self.tol or labels_unchanged:
        break

    self.centroids_ = centroids
    self.labels_ = labels
    return self

  def _WCSS(self, X, centroids, return_value=['error', 'distance']) :
    X = np.asarray(X)
    centroids = np.asarray(centroids)

    # Compute pairwise squared Euclidean distances
    diff = X[:, np.newaxis] - centroids
    distances = np.sum(diff ** 2, axis=2)

    labels = np.argmin(distances, axis=1).astype(np.int8)
    point_distance = np.min(distances, axis=1).astype(np.float16)
    error = np.sum(point_distance)

    output_map = {
        'error': error,
        'distance': point_distance,
        'labels': labels
    }

    results = [output_map[key] for key in return_value if key in output_map]
    return results[0] if len(results) == 1 else tuple(results)

  def _init_centroids(self, X) :
    X = np.asarray(X)

    # Validate that we have enough unique points to choose k centroids
    unique_points = np.unique(X, axis=0)
    if len(unique_points) < self.k :
      raise ValueError("Not enough unique points to choose k centroids")

    # K-Means++ initialization: pick first centroid randomly
    indices = list(np.random.choice(X.shape[0], 1))

    # Select remaining centroids based on distance to nearest existing centroid
    for ctr in range(self.k - 1) :
      _, distance = self._WCSS(X, X[indices])

      sum_dist = np.sum(distance)
      if sum_dist == 0:
        probabilities = np.ones(X.shape[0]) / X.shape[0]
      else:
        probabilities = distance / sum_dist

      idx = np.random.choice(X.shape[0], p=probabilities)
      indices.append(idx)

    centroids = X[indices].copy()
    return centroids

  def predict(self, X) :
    X = np.asarray(X)
    _, distance = self._WCSS(X, self.centroids_, return_value=['distance', 'labels'])
    return distance, self.labels_
