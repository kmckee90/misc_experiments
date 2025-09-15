import torch


class TorchKMeans:
    def __init__(self, n_clusters, max_iter=10, min_size=5, tol=1e-2):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.min_size = min_size

        self.centroids = None

    def fit(self, data):
        # Randomly initialize centroids
        sub = data[: self.n_clusters]
        self.centroids = data.mean() + torch.randn_like(sub) * 1e-2
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


class TorchKMeans2:
    def __init__(self, n_clusters, max_iter=10, min_size=5, tol=1e-2):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.min_size = min_size

        self.centroids = None

    def fit(self, data):
        # Randomly initialize centroids
        sub = data[: self.n_clusters]
        self.centroids = sub

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

        # Modify this to try
        if len(self.n_clusters) == 1:
            kmeans = TorchKMeans2(
                n_clusters[0],
                max_iter=self.max_iter,
                min_size=self.minimum_cluster_size,
            )
            fit = kmeans.fit(data)
            print(f"Done fitting: {fit}")
        else:
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
        # return -self.ExpNorm(Y, x, beta=200)
        # return -torch.cosine_similarity(Y, x.reshape(1, -1), dim=1)
        # return torch.norm(x - Y, dim=1)
        return torch.cdist(Y, x.unsqueeze(0)).squeeze()
        # return -Y @ x

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
    import time

    import matplotlib.pyplot as plt
    import numpy as np

    with torch.no_grad():
        device = "cuda:3"
        # Data:
        seq_len = int(1e4)
        n = 2

        n_means = 15
        means = torch.randn(n_means, n, device=device)  # Example data: 100 samples, 5 features each
        means = means[torch.randint(0, n_means, (seq_len,))]
        data = means + 0.25 * torch.randn(seq_len, n, device=device)  # Example data: 100 samples, 5 features each

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
        plot_tree(mkmeans.root)
        plt.savefig("test_cluster_fig.png")
        plt.close()
        print_tree(mkmeans.root)

        # Model:
        n_clusters = [8]  # Example: 3 clusters at the first level, then 2 clusters within each of those
        mkmeans = MultiKMeans(
            n_clusters=n_clusters,
            max_iter=5000,
            fit_threshold=-float("inf"),
            minimum_cluster_size=5,
            device=device,
        )

        # Fit and examine
        mkmeans.fit(data)
        print_tree(mkmeans.root)

        # Examine result
        plot_tree(mkmeans.root)
        plt.savefig("test_cluster_fig2.png", dpi=300)
        plt.close()

        # Test recall error
        n_samples = 1000
        acc_tree = 0
        acc_dot = 0
        for _ in range(n_samples):
            x = data[torch.randint(0, seq_len, (1,))].squeeze()
            x_hat = x.clone()
            x_hat += torch.randn_like(x_hat) * 0.1
            # x_hat[:16] = 0
            m = mkmeans.recall(x_hat)
            # Compare with usual dot prod
            m_dp = data[torch.argmax(torch.cosine_similarity(data, x_hat.reshape(1, -1)))]
            acc_tree += torch.corrcoef(torch.stack((x, m)))[1, 0] > 0.999
            acc_dot += torch.corrcoef(torch.stack((x, m_dp)))[1, 0] > 0.999

        acc_tree = acc_tree / n_samples
        acc_dot = acc_dot / n_samples
        acc_tree, acc_dot

        t0 = time.time()
        a = data[torch.argmax(torch.cosine_similarity(data, x))]
        time_dot = time.time() - t0

        t0 = time.time()
        similar_vector = mkmeans.recall(x)
        time_tree = time.time() - t0

    with torch.no_grad():
        device = "cuda:3"
        n_samples = 50

        output_vec_size_2 = {"dot": [], "tree": []}
        vec_sizes = [16, 32, 64, 128, 256, 512, 1024]
        seq_len = 1e6
        for n in vec_sizes:
            print(n)
            n_clusters = [2] * 2  # Example: 3 clusters at the first level, then 2 clusters within each of those
            mkmeans = MultiKMeans(n_clusters=n_clusters, max_iter=5, device=device)
            data = torch.randn(int(seq_len), n, device=device)  # Example data: 100 samples, 5 features each
            mkmeans.fit(data)
            samples_tree = []
            samples_dot = []
            for s in range(n_samples):
                print(s)
                x = torch.randn(1, n, device=device)
                t0 = time.time()
                a = data[torch.argmax(torch.cosine_similarity(data, x))]
                time_dot = time.time() - t0

                t0 = time.time()
                similar_vector = mkmeans.recall(x)
                time_tree = time.time() - t0

                samples_tree.append(time_tree)
                samples_dot.append(time_dot)

            output_vec_size_2["dot"].append(sum(samples_dot) / n_samples)
            output_vec_size_2["tree"].append(sum(samples_tree) / n_samples)

            del data
            del mkmeans

        output_vec_size_4 = {"dot": [], "tree": []}
        for n in vec_sizes:
            print(n)
            n_clusters = [2] * 4  # Example: 3 clusters at the first level, then 2 clusters within each of those
            mkmeans = MultiKMeans(n_clusters=n_clusters, max_iter=5, device=device)
            data = torch.randn(int(seq_len), n, device=device)  # Example data: 100 samples, 5 features each
            mkmeans.fit(data)
            samples_tree = []
            samples_dot = []
            for s in range(n_samples):
                print(s)
                x = torch.randn(1, n, device=device)
                t0 = time.time()
                a = data[torch.argmax(torch.cosine_similarity(data, x))]
                time_dot = time.time() - t0

                t0 = time.time()
                similar_vector = mkmeans.recall(x)
                time_tree = time.time() - t0

                samples_tree.append(time_tree)
                samples_dot.append(time_dot)

            output_vec_size_4["dot"].append(sum(samples_dot) / n_samples)
            output_vec_size_4["tree"].append(sum(samples_tree) / n_samples)

            del data
            del mkmeans

        output_vec_size_6 = {"dot": [], "tree": []}
        for n in vec_sizes:
            print(n)
            n_clusters = [2] * 6  # Example: 3 clusters at the first level, then 2 clusters within each of those
            mkmeans = MultiKMeans(n_clusters=n_clusters, max_iter=5, device=device)
            data = torch.randn(int(seq_len), n, device=device)  # Example data: 100 samples, 5 features each
            mkmeans.fit(data)
            samples_tree = []
            samples_dot = []
            for s in range(n_samples):
                print(s)
                x = torch.randn(1, n, device=device)
                t0 = time.time()
                a = data[torch.argmax(torch.cosine_similarity(data, x))]
                time_dot = time.time() - t0

                t0 = time.time()
                similar_vector = mkmeans.recall(x)
                time_tree = time.time() - t0

                samples_tree.append(time_tree)
                samples_dot.append(time_dot)

            output_vec_size_6["dot"].append(sum(samples_dot) / n_samples)
            output_vec_size_6["tree"].append(sum(samples_tree) / n_samples)

            del data
            del mkmeans

        output_seq_len_2 = {"dot": [], "tree": []}
        seq_lens = [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6]
        vec_size = 256
        for n in seq_lens:
            print(n)
            n_clusters = [2] * 2  # Example: 3 clusters at the first level, then 2 clusters within each of those
            mkmeans = MultiKMeans(n_clusters=n_clusters, max_iter=5, device=device)
            data = torch.randn(int(n), vec_size, device=device)  # Example data: 100 samples, 5 features each
            mkmeans.fit(data)
            samples_tree = []
            samples_dot = []
            for s in range(n_samples):
                print(s)
                x = torch.randn(1, vec_size, device=device)
                t0 = time.time()
                a = data[torch.argmax(torch.cosine_similarity(data, x))]
                time_dot = time.time() - t0

                t0 = time.time()
                similar_vector = mkmeans.recall(x)
                time_tree = time.time() - t0

                samples_tree.append(time_tree)
                samples_dot.append(time_dot)

            output_seq_len_2["dot"].append(sum(samples_dot) / n_samples)
            output_seq_len_2["tree"].append(sum(samples_tree) / n_samples)

            del data
            del mkmeans

        output_seq_len_4 = {"dot": [], "tree": []}
        vec_size = 256
        for n in seq_lens:
            print(n)
            n_clusters = [2] * 4  # Example: 3 clusters at the first level, then 2 clusters within each of those
            mkmeans = MultiKMeans(n_clusters=n_clusters, max_iter=5, device=device)
            data = torch.randn(int(n), vec_size, device=device)  # Example data: 100 samples, 5 features each
            mkmeans.fit(data)
            samples_tree = []
            samples_dot = []
            for s in range(n_samples):
                print(s)
                x = torch.randn(1, vec_size, device=device)
                t0 = time.time()
                a = data[torch.argmax(torch.cosine_similarity(data, x))]
                time_dot = time.time() - t0

                t0 = time.time()
                similar_vector = mkmeans.recall(x)
                time_tree = time.time() - t0

                samples_tree.append(time_tree)
                samples_dot.append(time_dot)

            output_seq_len_4["dot"].append(sum(samples_dot) / n_samples)
            output_seq_len_4["tree"].append(sum(samples_tree) / n_samples)

            del data
            del mkmeans

        output_seq_len_6 = {"dot": [], "tree": []}
        for n in seq_lens:
            print(n)
            n_clusters = [2] * 6  # Example: 3 clusters at the first level, then 2 clusters within each of those
            mkmeans = MultiKMeans(n_clusters=n_clusters, max_iter=5, device=device)
            data = torch.randn(int(n), vec_size, device=device)  # Example data: 100 samples, 5 features each
            mkmeans.fit(data)
            samples_tree = []
            samples_dot = []
            for s in range(n_samples):
                print(s)
                x = torch.randn(1, vec_size, device=device)
                t0 = time.time()
                a = data[torch.argmax(torch.cosine_similarity(data, x))]
                time_dot = time.time() - t0

                t0 = time.time()
                similar_vector = mkmeans.recall(x)
                time_tree = time.time() - t0

                samples_tree.append(time_tree)
                samples_dot.append(time_dot)

            output_seq_len_6["dot"].append(sum(samples_dot) / n_samples)
            output_seq_len_6["tree"].append(sum(samples_tree) / n_samples)

            del data
            del mkmeans

    plt.plot(np.log2(vec_sizes), np.log10(output_vec_size_2["dot"]))
    plt.plot(np.log2(vec_sizes), np.log10(output_vec_size_2["tree"]))
    plt.plot(np.log2(vec_sizes), np.log10(output_vec_size_4["tree"]))
    plt.plot(np.log2(vec_sizes), np.log10(output_vec_size_6["tree"]))

    # Adding a title
    plt.title("Memory performance comparison: vector size")

    # Adding X and Y axis labels
    plt.xlabel("Log2 Vector Size | Seq. len 1e6")
    plt.ylabel("Log10 Recall time (s)")

    # Showing legend
    plt.savefig("linegraph_vec_size.png")
    plt.close()

    plt.plot(np.log10(seq_lens), np.log10(output_seq_len_4["dot"]))
    plt.plot(np.log10(seq_lens), np.log10(output_seq_len_2["tree"]))
    plt.plot(np.log10(seq_lens), np.log10(output_seq_len_4["tree"]))
    plt.plot(np.log10(seq_lens), np.log10(output_seq_len_6["tree"]))

    # Adding a title
    plt.title("Memory performance comparison: sequence length")

    # Adding X and Y axis labels
    plt.xlabel("Log10 Sequence Length | Vec. size. 256")
    plt.ylabel("Log10 Recall time (s)")

    # Showing legend
    plt.savefig("linegraph_seq_len.png")
    plt.close()
