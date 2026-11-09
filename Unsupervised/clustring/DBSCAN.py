class Custom_DBSCAN() :
  def __init__(self, epsilon, min_samples) :
    if epsilon <= 0:
      raise ValueError("eps must be greater than 0.")

    if not isinstance(min_samples, (int, np.integer)):
      raise TypeError("min_samples must be an integer.")

    if min_samples < 1:
      raise ValueError("min_samples must be at least 1.")

    self.epsilon = epsilon
    self.min_samples = min_samples
    self.labels_ = None

  def region_query(self, point, X) :
    X = np.asarray(X)
    pt = X[point]

    distance = np.linalg.norm(X - pt, axis=1)

    return np.flatnonzero(distance <= self.epsilon)

  def fit_predict(self, X) :
    X = np.asarray(X)

    if X.ndim != 2:
      raise ValueError("X must be a 2D array.")

    if len(X) == 0:
      raise ValueError("X must not be empty.")

    if not np.all(np.isfinite(X)):
      raise ValueError("X must contain only finite values.")

    visited = set()
    clusters = []
    noise = set()
    labels = np.zeros(X.shape[0], dtype=int)

    for pt in range(X.shape[0]) :
      if pt in visited :
        continue

      visited.add(pt)
      neighbour = self.region_query(pt, X)

      if len(neighbour) < self.min_samples :
        noise.add(pt)
        continue

      clustr = []
      clustr.append(pt)
      queue = deque(neighbour)

      while len(queue) != 0:
          q = queue.popleft()

          # A previously identified noise point may be a border point.
          if q in noise:
              noise.remove(q)
              clustr.append(q)
              continue

          if q in visited:
              continue

          visited.add(q)
          q_neighnour = self.region_query(q, X)

          if len(q_neighnour) >= self.min_samples:
              queue.extend(q_neighnour)

          clustr.append(q)

      clusters.append(clustr)

    for idx in range(len(clusters)) :
      index = clusters[idx]
      labels[index] = idx

    for i in noise :
      labels[i] = -1

    self.labels_ = labels

    return labels
