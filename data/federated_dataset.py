import numpy as np
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, Optional, List, Literal
from scipy.stats import entropy
from tqdm import tqdm


class FederatedDataSplitter:
    """
    Class to split dataset indices for federated learning experiments.

    This class computes and stores the indices for each client and the global model,
    according to the specified sampling strategy.

    Args:
        dataset (Dataset): The base dataset to be split.
        num_clients (int): The total number of clients in the federated setup.
        sampling_strategy (str): The strategy for splitting the data. Options are
                                 "stratified", "random", "iid", or "non_iid".
        alpha (float): The alpha parameter for the Dirichlet distribution in non-IID sampling.
        get_label_fn (Callable): A function to extract the label from a dataset sample.
        global_fraction (float): Fraction of the data to be used in the global dataset.
                                 Must be between 0 and 1. Default is 1.0 (all data).
    """

    def __init__(
        self,
        dataset: Dataset,
        num_clients: int,
        sampling_strategy: Literal[
            "stratified", "random", "iid", "non_iid"
        ] = "stratified",
        alpha: float = 1.0,
        get_label_fn: Optional[Callable] = None,
        global_fraction: float = 0.5,  # Default to 0.0 for disjoint datasets
        random_seed: Optional[int] = None,
        client_proportions: Optional[List[float]] = None,
    ):
        if random_seed is not None:
            np.random.seed(int(random_seed))
        self.dataset = dataset
        self.num_clients = num_clients
        self.sampling_strategy = sampling_strategy
        self.alpha = alpha
        self.get_label_fn = get_label_fn
        self.global_fraction = global_fraction

        if client_proportions is not None:
            self.client_proportions = client_proportions
        else:
            self.client_proportions = [1 / num_clients] * num_clients

        self.all_indices = np.arange(len(self.dataset))
        self.client_indices = {}  # Dict[client_id, indices]
        self.global_indices = None  # Indices for global dataset

        self._partition_data()
        self._split_data()

    def _partition_data(self):
        """
        Partition the data into global and client datasets based on global_fraction.
        """
        num_global_samples = int(len(self.all_indices) * self.global_fraction)
        if num_global_samples > 0:
            self.global_indices = np.random.choice(
                self.all_indices, size=num_global_samples, replace=False
            )
            # Remaining indices are for clients
            self.client_data_indices = np.setdiff1d(
                self.all_indices, self.global_indices
            )
        else:
            self.global_indices = np.array([])
            self.client_data_indices = self.all_indices.copy()

    def _split_data(self):
        """
        Split the client data according to the specified sampling strategy.
        """
        if self.sampling_strategy == "stratified":
            self._stratified_split()
        elif self.sampling_strategy == "random":
            self._random_split()
        elif self.sampling_strategy == "iid":
            self._iid_split()
        elif self.sampling_strategy == "non_iid":
            self._non_iid_split()
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def _stratified_split(self):
        """
        Perform a stratified split of the client data.
        """
        labels = [self.get_label_fn(self.dataset[i]) for i in self.client_data_indices]
        labels = np.array(labels)
        unique_labels = np.unique(labels)

        label_indices = {
            label: self.client_data_indices[np.where(labels == label)[0]]
            for label in unique_labels
        }

        for client_id in tqdm(
            range(1, self.num_clients + 1), desc="Stratified Split", ascii=True
        ):
            client_indices = []
            for label in unique_labels:
                indices = label_indices[label]
                splits = np.array_split(indices, self.num_clients)
                client_indices.extend(splits[client_id - 1])
            self.client_indices[client_id] = np.array(client_indices)

    def _random_split(self):
        """
        Perform a random split of the client data using custom client proportions.
        """
        shuffled_indices = self.client_data_indices.copy()
        np.random.shuffle(shuffled_indices)
        num_samples = len(shuffled_indices)
        proportions = (np.array(self.client_proportions) * num_samples).astype(int)

        # Adjust proportions to ensure sum equals num_samples
        diff = num_samples - np.sum(proportions)
        for i in range(abs(diff)):
            idx = i % self.num_clients
            proportions[idx] += np.sign(diff)

        split_indices = []
        start = 0
        for count in proportions:
            end = start + count
            split_indices.append(shuffled_indices[start:end])
            start = end

        for client_id in range(self.num_clients):
            self.client_indices[client_id] = split_indices[client_id]

    def _iid_split(self):
        """
        Perform an IID split of the client data.
        """
        self._random_split()  # IID split is equivalent to random split here

    def _non_iid_split(self):
        """
        Perform a non-IID split of the client data.
        """
        labels = [self.get_label_fn(self.dataset[i]) for i in self.client_data_indices]
        labels = np.array(labels)
        unique_labels = np.unique(labels)

        # For each class, partition its indices to clients based on Dirichlet distribution
        class_indices = {
            label: self.client_data_indices[np.where(labels == label)[0]]
            for label in unique_labels
        }
        client_class_indices = {client_id: [] for client_id in range(self.num_clients)}

        for label in unique_labels:
            indices = class_indices[label]
            # Sample a distribution over clients for this label
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
            # Split indices according to the proportions
            proportions = (proportions * len(indices)).astype(int)

            # Adjust proportions to ensure sum equals len(indices)
            diff = len(indices) - np.sum(proportions)
            for i in range(abs(diff)):
                idx = i % self.num_clients
                proportions[idx] += np.sign(diff)

            # Distribute indices to clients
            start = 0
            for client_id, count in enumerate(proportions):
                end = start + count
                client_class_indices[client_id].extend(indices[start:end])
                start = end

        for client_id in range(1, self.num_clients + 1):
            self.client_indices[client_id] = np.array(
                client_class_indices[client_id - 1]
            )

        print(self.client_indices.keys())

    def get_client_indices(self, client_id: int) -> np.array:
        """Get indices for a specific client."""
        return self.client_indices[client_id]

    def get_global_indices(self) -> np.array:
        """Get indices for the global dataset."""
        return self.global_indices

    def calculate_iid_metrics(self) -> Dict[str, float]:
        """
        Calculate metrics to measure the level of IID in the dataset split.

        Returns a dictionary with the following metrics:
        - label_distribution_divergence: Average Jensen-Shannon divergence of client
          label distributions from the overall distribution.
        - client_data_size_variance: Normalized variance in client data sizes.
        - average_kl_divergence: Average KL divergence of client label distributions
          from the overall distribution.

        Expected values:
        - IID scenario: All metrics very close to 0 (e.g., < 0.01)
        - Slight non-IID: Low values (e.g., 0.01 - 0.1)
        - Moderate non-IID: Medium values (e.g., 0.1 - 0.5)
        - Extreme non-IID: High values (e.g., > 0.5, potentially > 1.0 for KL divergence)

        Lower values indicate a more IID-like split. Higher values suggest more
        non-IID characteristics, which may lead to challenges in federated learning.

        Returns:
            Dict[str, float]: A dictionary containing the calculated metrics.
        """
        # Compute overall label distribution
        all_labels = [self.get_label_fn(self.dataset[i]) for i in self.all_indices]
        all_labels = np.array(all_labels)
        unique_labels = np.unique(all_labels)
        label_counts = np.bincount(all_labels, minlength=len(unique_labels))
        overall_distribution = label_counts / label_counts.sum()

        client_distributions = []
        client_sizes = []

        for client_id in range(1, self.num_clients + 1):
            client_indices = self.get_client_indices(client_id)
            client_labels = [self.get_label_fn(self.dataset[i]) for i in client_indices]
            client_labels = np.array(client_labels)
            client_label_counts = np.bincount(
                client_labels, minlength=len(unique_labels)
            )
            client_distribution = client_label_counts / client_label_counts.sum()
            client_distributions.append(client_distribution)
            client_sizes.append(len(client_indices))

        # 1. Label Distribution Divergence (using Jensen-Shannon divergence)
        js_divergences = [
            self._jensen_shannon_divergence(overall_distribution, cd)
            for cd in client_distributions
        ]
        avg_js_divergence = np.mean(js_divergences)

        # 2. Client Data Size Variance
        client_sizes = np.array(client_sizes)
        client_size_variance = np.var(client_sizes) / (np.mean(client_sizes) ** 2)

        # 3. Average KL Divergence
        kl_divergences = [
            entropy(cd, overall_distribution) for cd in client_distributions
        ]
        avg_kl_divergence = np.mean(kl_divergences)

        return {
            "label_distribution_divergence": avg_js_divergence,
            "client_data_size_variance": client_size_variance,
            "average_kl_divergence": avg_kl_divergence,
        }

    @staticmethod
    def _jensen_shannon_divergence(p: np.array, q: np.array) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        p = np.array(p)
        q = np.array(q)
        # Avoid division by zero and invalid values
        p = np.clip(p, 1e-12, 1)
        q = np.clip(q, 1e-12, 1)
        m = 0.5 * (p + q)
        return 0.5 * (entropy(p, m)) + 0.5 * (entropy(q, m))


class FederatedDatasetWrapper(Dataset):
    """
    A wrapper class for creating federated datasets from any PyTorch Dataset.

    This class uses precomputed indices to provide the dataset subset for a specific client or global model.

    Args:
        dataset (Dataset): The base dataset.
        indices (np.array): Indices for this client's dataset subset.
        get_label_fn (Callable): A function to extract the label from a dataset sample.

    Usage:
        base_dataset = YourDataset(...)
        splitter = FederatedDataSplitter(
            dataset=base_dataset,
            num_clients=10,
            sampling_strategy="stratified",
            get_label_fn=lambda x: x[1],
            global_fraction=0.5  # Use 50% of data for global model
        )
        global_dataset = FederatedDatasetWrapper(
            dataset=base_dataset,
            indices=splitter.get_global_indices(),
            get_label_fn=lambda x: x[1]
        )
        client_datasets = []
        for client_id in range(10):
            client_indices = splitter.get_client_indices(client_id)
            client_dataset = FederatedDatasetWrapper(
                dataset=base_dataset,
                indices=client_indices,
                get_label_fn=lambda x: x[1]
            )
            client_datasets.append(client_dataset)
    """

    def __init__(
        self,
        dataset: Dataset,
        indices: np.array,
        get_label_fn: Optional[Callable] = None,
    ) -> "FederatedDatasetWrapper":
        self.base_dataset = dataset
        self.indices = indices
        self.get_label_fn = get_label_fn or (lambda x: x[1])

    def __len__(self) -> int:
        """
        Get the length of this dataset subset.

        Returns:
            int: The number of samples in this subset.
        """
        return len(self.indices)

    def __getitem__(self, idx) -> Any:
        """
        Get an item from this dataset subset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Any: The dataset item at the specified index.
        """
        return self.base_dataset[self.indices[idx]]

    def get_labels(self) -> list:
        """
        Get all labels for this dataset subset.

        Returns:
            list: A list of labels for all samples in this subset.
        """
        return [self.get_label_fn(self.base_dataset[i]) for i in self.indices]

    def collate_fn(self, batch) -> Any:
        """
        Collate function for creating batches.

        If the base dataset has a collate_fn method, it will be used.
        Otherwise, the default PyTorch collate function will be used.

        Args:
            batch (list): A list of samples to be collated into a batch.

        Returns:
            Any: The collated batch.
        """
        if (
            hasattr(self.base_dataset, "collate_fn")
            and self.base_dataset.collate_fn is not None
        ):
            return self.base_dataset.collate_fn(batch)
        return None  # Default PyTorch collate_fn will be used
