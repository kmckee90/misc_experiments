import matplotlib.pyplot as plt
import torch


class TorchKMeans:
    def __init__(self, n_clusters, max_iter=10, min_size=5, tol=1e-2):
        """K-Means clustering, specifically for 2 class clustering in binary trees.
        Returns a split criterion based on the mahalanobis distance of each centroid to the other."""
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.min_size = min_size

        self.centroids = None

    def fit(self, data):
        # Randomly initialize centroids
        idx = torch.randperm(data.size(0))[: self.n_clusters]
        self.centroids = data[idx]

        for _ in range(self.max_iter):
            # E-step: assign clusters
            distances = torch.cdist(data, self.centroids, p=2)
            labels = torch.argmin(distances, dim=1)

            # M-step: update centroids
            new_centroids = torch.stack([data[labels == i].mean(dim=0) for i in range(self.n_clusters)])

            # Check for convergence
            delta = torch.norm(self.centroids - new_centroids, dim=1).max()
            if delta < self.tol:
                break
            self.centroids = new_centroids
            self.labels_ = labels

        # Return poor fit if any cluster is too small
        size_check = torch.tensor([len(data[labels == i]) <= self.min_size for i in range(self.n_clusters)])
        if any(size_check):
            return 0

        # Use mahalanobis distance
        covs = [torch.cov(data[labels == i].T) for i in range(self.n_clusters)]
        distance0 = mahalanobis(self.centroids[:1], self.centroids[1:], covs[1])
        distance1 = mahalanobis(self.centroids[1:], self.centroids[:1], covs[0])
        criterion = torch.tensor([distance0, distance1]).min().item()
        # criterion = torch.tensor([distance0, distance1]).min().item()
        return criterion  # Return as a Python float for easier interpretation


def mahalanobis(u, v, cov):
    delta = u - v
    m = torch.sqrt(delta @ torch.inverse(cov) @ delta.T)
    return m


class MultiKMeans:
    def __init__(
        self,
        n_clusters,
        max_iter,
        minimum_cluster_size=5,
        fit_threshold=1,
        device="cuda",
    ):
        """
        Initialize the MultiKMeans class with a hierarchy of clusters.

        :param n_clusters: List of integers specifying the number of clusters at each level.
        :param device: The PyTorch device ('cuda' or 'cpu') the calculations should run on.
        """
        self.n_clusters = n_clusters
        self.device = device
        self.max_iter = max_iter
        self.fit_threshold = fit_threshold
        self.minimum_cluster_size = minimum_cluster_size
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

        kmeans = TorchKMeans(
            n_clusters[0],
            max_iter=self.max_iter,
            min_size=self.minimum_cluster_size,
        )
        fit = kmeans.fit(data)
        print(f"Done fitting: {fit}")
        if fit < self.fit_threshold:
            return {"centroid": None, "data": data}

        centroids = kmeans.centroids

        clusters = []
        for i in range(n_clusters[0]):
            cluster_data = data[kmeans.labels_ == i]
            clusters.append(self._fit_recursive(cluster_data, n_clusters[1:]))

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
            distances = self.dist(node["centroid"], vector)
            closest = torch.argmin(distances)
            _store_recursive(node["sub_clusters"][closest], vector)

        _store_recursive(self.root, vector)

    def recall(self, vector):
        vector = vector.to(self.device)

        def _recall_recursive(node, vector):
            if node["centroid"] is None:
                distances = self.dist(node["data"], vector)
                closest_idx = torch.argmin(distances)
                return node["data"][closest_idx]
            distances = self.dist(node["centroid"], vector)
            closest = torch.argmin(distances)
            return _recall_recursive(node["sub_clusters"][closest], vector)

        return _recall_recursive(self.root, vector)

    def dist(self, Y, x):
        return torch.cdist(Y, x.unsqueeze(0)).squeeze()

    def ExpNorm(self, Y, x, beta=80):  # Subset similarity
        Z = (x - Y) ** 2
        out = torch.mean(torch.exp(-Z * beta), 1)
        return out


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


def plot_tree(node, level=0):
    """
    Recursively prints the tree structure of the hierarchical k-means clustering.

    :param node: The current node in the tree.
    :param level: The current level in the tree (used for indentation).
    """
    # Check if the current node is a leaf node (no sub-clusters)
    if "centroid" in node and node["centroid"] is not None:
        print("  " * level + f"Level {level}: Centroid")
    else:
        print("  " * level + f"Level {level}: Leaf Node, Length {len(node['data'])}")
        plot_data = node["data"].cpu().numpy()
        plt.scatter(plot_data[:, 0], plot_data[:, 1], s=1)

    # If the current node has sub-clusters, recursively print each sub-cluster
    if "sub_clusters" in node:
        for sub_cluster in node["sub_clusters"]:
            plot_tree(sub_cluster, level + 1)


if __name__ == "__main__":
    with torch.no_grad():
        device = "cuda:3"
        # Data:
        seq_len = int(1e4)
        n = 2

        # n_means = 15
        # means = torch.randn(n_means, n, device=device)  # Example data: 100 samples, 5 features each
        # means = means[torch.randint(0, n_means, (seq_len,))]
        # data = means + 0.25 * torch.randn(seq_len, n, device=device)  # Example data: 100 samples, 5 features each

        # # # Another method of generating data with random clustering: random walk
        data = []
        s = 5
        x = s * torch.randn(s, n, device=device)
        for _ in range(5000):
            data.append(x)
            x = x + torch.randn(s, n, device=device)
        data = torch.cat(data)

        data = data / data.std(0)
        data = data[torch.randperm(len(data))]

        # Model:
        n_clusters = [2] * 14  # Example: 3 clusters at the first level, then 2 clusters within each of those
        mkmeans = MultiKMeans(
            n_clusters=n_clusters,
            max_iter=500,
            fit_threshold=2.9,
            minimum_cluster_size=100,
            device=device,
        )

        # Fit and examine
        mkmeans.fit(data)

        # Examine result
        print_tree(mkmeans.root)

        plot_tree(mkmeans.root)
        plt.savefig("test_cluster_fig.png")
        plt.close()
