import time

import torch


def print_tree(node, level=0):
    """
    Recursively prints the tree structure of the hierarchical k-means clustering.

    :param node: The current node in the tree.
    :param level: The current level in the tree (used for indentation).
    """
    n_clusters = 0
    # Check if the current node is a leaf node (no sub-clusters)
    if "centroid" in node and node["centroid"] is not None:
        print("  " * level + f"Level {level}: Centroid")
    else:
        print("  " * level + f"Level {level}: Leaf Node, Length {len(node['data'])}")
        n_clusters = 1

    # If the current node has sub-clusters, recursively print each sub-cluster
    if "sub_clusters" in node:
        for sub_cluster in node["sub_clusters"]:
            n_clusters += print_tree(sub_cluster, level + 1)

    if level == 0:
        print("Number of clusters:", n_clusters)

    return n_clusters


class TorchKMeans:
    def __init__(self, n_clusters, max_iter=10, tol=1e-2):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, data):
        # Randomly initialize centroids
        idx = torch.randperm(data.size(0))[: self.n_clusters]
        self.centroids = data[idx]

        for _ in range(self.max_iter):
            # E-step: assign clusters
            distances = torch.cdist(data, self.centroids)
            labels = torch.argmin(distances, dim=1)

            # M-step: update centroids
            new_centroids = torch.stack([data[labels == i].mean(dim=0) for i in range(self.n_clusters)])

            # Check for convergence
            delta = torch.norm(self.centroids - new_centroids, dim=1).max()
            print(delta)
            if delta < self.tol:
                break
            self.centroids = new_centroids
            self.labels_ = labels


class MultiKMeans:
    def __init__(self, n_clusters, device="cuda"):
        """
        Initialize the MultiKMeans class with a hierarchy of clusters.

        :param n_clusters: List of integers specifying the number of clusters at each level.
        :param device: The PyTorch device ('cuda' or 'cpu') the calculations should run on.
        """
        self.n_clusters = n_clusters
        self.device = device
        self.root = {}

    def fit(self, data):
        """
        Fit the hierarchical k-means clustering model on the data.

        :param data: A PyTorch tensor of shape (n_samples, n_features) with the input data.
        """
        data = data.to(self.device)
        self.root = self._fit_recursive(data, self.n_clusters)

    def _fit_recursive(self, data, n_clusters):
        if len(n_clusters) == 0:
            return {"centroid": None, "data": data}

        kmeans = TorchKMeans(n_clusters[0])
        kmeans.fit(data)
        centroids = kmeans.centroids

        clusters = []
        for i in range(n_clusters[0]):
            cluster_data = data[kmeans.labels_ == i]
            clusters.append(self._fit_recursive(cluster_data, n_clusters[1:]))

        print("Done fitting")
        return {"centroid": centroids, "sub_clusters": clusters}

    def store(self, vector):
        vector = vector.to(self.device)

        def _store_recursive(node, vector):
            if node["centroid"] is None:
                if "data" in node:
                    node["data"] = torch.cat([node["data"], vector.unsqueeze(0)], dim=0)
                else:
                    node["data"] = vector.unsqueeze(0)
                return
            distances = torch.norm(vector - node["centroid"], dim=1)
            closest = torch.argmin(distances)
            _store_recursive(node["sub_clusters"][closest], vector)

        _store_recursive(self.root, vector)

    def recall(self, vector):
        vector = vector.to(self.device)

        def _recall_recursive(node, vector):
            if node["centroid"] is None:
                distances = torch.norm(node["data"] - vector, dim=1)
                closest_idx = torch.argmin(distances)
                return node["data"][closest_idx]

            distances = torch.norm(vector - node["centroid"], dim=1)
            closest = torch.argmin(distances)
            return _recall_recursive(node["sub_clusters"][closest], vector)

        return _recall_recursive(self.root, vector)


with torch.no_grad():
    device = "cuda:3"
    n = 256
    seq_len = 10000
    # Example usage
    n_clusters = [2] * 6  # Example: 3 clusters at the first level, then 2 clusters within each of those
    mkmeans = MultiKMeans(n_clusters=n_clusters, device=device)
    data = torch.randn(10000, n, device=device)  # Example data: 100 samples, 5 features each

    mkmeans.n_clusters = [2] * 6
    mkmeans.fit(data)
    vector_to_store = torch.randn(n, device=device)
    mkmeans.store(vector_to_store)
    similar_vector = mkmeans.recall(vector_to_store)

    vector_to_store = torch.randn(n, device=device)
    similar_vector = mkmeans.recall(vector_to_store)
    torch.corrcoef(torch.stack((vector_to_store, similar_vector)))

    x = torch.randn(1, n, device=device)

    t0 = time.time()
    a = data[torch.argmax(torch.cosine_similarity(data, x))]
    print(time.time() - t0)

    t0 = time.time()
    similar_vector = mkmeans.recall(x)
    print(time.time() - t0)

    print_tree(mkmeans.root)
