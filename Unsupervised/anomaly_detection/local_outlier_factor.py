class Local_Outlier_Factor() :
    """
    A custom implementation of the Local Outlier Factor (LOF) algorithm
    used for identifying anomalous spatial data points based on local density.
    """
    def __init__(self, k, threshold=None) :
        # k represents the number of nearest neighbors to consider
        self.k = k
        self.threshold = threshold

        self.reachability_distance = None
        self.k_distance = None
        self.local_density = None
        self.neighbor = {}

    def fit(self, X) :
        """
        Computes pairwise distances, determines k-distances and neighbors for each point,
        and calculates local reachability densities.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        self.k_distance = []

        # Compute pairwise Euclidean distances between all samples
        distance = cdist(X, X, metric="euclidean")
        # Set self-distance to infinity so a point isn't selected as its own neighbor
        np.fill_diagonal(distance, val=np.inf)

        # Determine k-nearest neighbors and the corresponding k-distance for each point
        for idx in range(n_samples) :
            # Sort indices of distances in ascending order and take top k closest
            neighbor_idx = np.argsort(distance[idx, :])[:self.k]
            self.neighbor[idx] = neighbor_idx

            # k-distance is the distance to the k-th nearest neighbor
            self.k_distance.append(distance[idx][neighbor_idx[self.k - 1]])

        reach_dist = np.zeros((n_samples, self.k))
        loc_den = np.zeros(n_samples)

        # Calculate reachability distances and local reachability density (lrd)
        for pt, neig in self.neighbor.items() :
            dist = []
            for o in neig :
                # Reachability distance of pt with respect to neighbor o is:
                # max( k-distance of o, actual Euclidean distance between pt and o )
                reachability = max(self.k_distance[o], distance[pt][o])
                dist.append(reachability)

            reach_dist[pt] = dist
            # Local reachability density is the inverse of the average reachability distance
            loc_den[pt] = 1 / (self.k * np.sum(dist))

        self.reachability_distance = reach_dist
        self.local_density = loc_den

        return self

    def lof(self) :
        """
        Computes the final Local Outlier Factor score for each point.
        LOF is the average ratio of the local reachability density of its neighbors to its own.
        """
        if self.local_density is None :
            raise ValueError("Fit the data before calculating LOF scores.")

        outlier_factor = []

        # Compute the LOF score for each data point
        for pt, neig in self.neighbor.items() :
            pt_den = self.local_density[pt]
            x = 0
            for o in neig :
                # Sum up the ratio of neighbor local densities to this point's density
                x += self.local_density[o] / pt_den

            # LOF is the average density ratio of its neighbors
            outlier_factor.append((1 / self.k) * x)

        return outlier_factor
