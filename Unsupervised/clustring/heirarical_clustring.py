from shared import np, cdist

class Hierarchical_Clustering() :
  """
  A custom implementation of Agglomerative Hierarchical Clustering.
  Supports Single, Complete, Average, and Ward linkage methods.
  """
  def __init__(self, linkage='single', n_estimators=2, metric='euclidean') :
    self.linkage = linkage
    self.n_estimators = n_estimators
    self.metric = metric
    self.merge_history_ = None
    self.labels_ = None

  def fit(self, X) :
    X = np.asarray(X)
    n = X.shape[0]

    # Keep track of active cluster IDs and their actual member indices
    active = set(i for i in range(n))
    clusters = {idx: {idx} for idx in range(n)}
    self.merge_history_ = []

    # Distance matrix holds distances for initial samples and all newly merged parent clusters
    distances = np.full((2 * n - 1, 2 * n - 1), np.inf)
    distances[:n, :n] = cdist(X, X, metric=self.metric)
    np.fill_diagonal(distances, np.inf)

    previous_active_state = len(active)
    merge_count = 0

    # Main clustering loop: merge clusters until only one remains
    while len(active) > 1 :
      # 1. Find the two closest active clusters
      idx = np.argmin(distances)
      a, b = np.unravel_index(idx, distances.shape)
      merge_height = distances[a][b]

      # 2. Register the newly merged parent cluster
      new_id = n + merge_count
      clusters[new_id] = clusters[a] | clusters[b]
      self.merge_history_.append((a, b, merge_height, len(clusters[new_id])))

      # 3. Calculate distance from the new parent cluster to all other active clusters
      remaining = active - {a, b}
      for i_id in remaining :
        d_ai = distances[a][i_id]
        d_bi = distances[b][i_id]

        # Reduced clutter: Clean and direct mapping of the linkage calculations
        if self.linkage == 'single' :
            d = min(d_ai, d_bi)
        elif self.linkage == 'complete' :
            d = max(d_ai, d_bi)
        elif self.linkage == 'average' :
            A, B = len(clusters[a]), len(clusters[b])
            d = (A * d_ai + B * d_bi) / (A + B)
        elif self.linkage == 'ward' :
            if self.metric != 'euclidean' :
              raise ValueError("Ward linkage requires Euclidean distance")
            n_r = len(clusters[i_id])
            n_a, n_b = len(clusters[a]), len(clusters[b])
            T = n_r + n_a + n_b
            # Lance-Williams update formula for Ward's method
            d_ar = ((n_r + n_a) / T) * (d_ai ** 2)
            d_br = ((n_r + n_b) / T) * (d_bi ** 2)
            d_ab = ((n_r) / T) * (distances[a][b] ** 2)
            d = np.sqrt(d_ar + d_br - d_ab)
        else:
            raise ValueError(f"Invalid linkage method: {self.linkage}")

        distances[new_id, i_id] = d
        distances[i_id, new_id] = d

      # 4. Deactivate the merged children in the distance matrix
      distances[a, :] = np.inf
      distances[:, a] = np.inf
      distances[b, :] = np.inf
      distances[:, b] = np.inf

      active.difference_update({a, b})
      active.add(new_id)

      # Verification checks
      assert len(active) == previous_active_state - 1
      assert np.array_equal(distances, distances.T)
      previous_active_state = len(active)
      merge_count += 1

    assert len(self.merge_history_) == n - 1
    assert len(active) == 1

    # Assign final cluster labels by cutting the dendrogram tree
    self.labels_ = np.zeros(n, dtype=int)
    final_clusters = self._cut_tree()

    for label, cl in enumerate(final_clusters):
      self.labels_[list(clusters[cl])] = label

    return self

  def _cut_tree(self) :
    """Cut the dendrogram at the step corresponding to the requested number of clusters."""
    n = len(self.merge_history_) + 1
    merges_req = n - self.n_estimators
    merges = self.merge_history_[:merges_req]

    cluster_id = set(i for i in range(n))
    for idx in range(merges_req) :
      if (merges[idx][0] not in cluster_id) and (merges[idx][1] not in cluster_id):
        cluster_id.add(idx + n)
      else:
        cluster_id.discard(merges[idx][0])
        cluster_id.discard(merges[idx][1])
        cluster_id.add(idx + n)

    return list(cluster_id)
